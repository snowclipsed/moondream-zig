const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;
const math = std.math;

/// SlabReusingAllocator wraps another allocator and maintains a cache of recently freed allocations
/// organized by size class. This reduces the number of mmaps/munmaps for programs that
/// repeatedly allocate and free memory of the same sizes.
pub fn SlabReusingAllocator(comptime DEQUE_SIZE: usize) type {
    const PAGE_ALIGN = 4096;
    const THRESHOLD = 2048; // Based on GPA's largest_bucket_object_size (page_size / 2)

    return struct {
        const Self = @This();

        // We'll use one slot for each power of 2 (up to 64)
        const NUM_BUCKETS = 64;

        backing_allocator: Allocator,

        // Our caches of free slabs, organized by size class
        // For each bucket, we have a deque implemented as a circular buffer
        slabs: [NUM_BUCKETS][DEQUE_SIZE]?[*]align(PAGE_ALIGN) u8,
        tops: [NUM_BUCKETS]usize,  // Index of the top element + 1 (0 means empty)
        sizes: [NUM_BUCKETS]usize, // Number of elements in the deque

        // Tracks the original size of each allocation for later freeing
        size_map: std.AutoHashMap(usize, usize),

        pub fn init(backing_allocator: Allocator) Self {
            const self = Self{
                .backing_allocator = backing_allocator,
                .slabs = [_][DEQUE_SIZE]?[*]align(PAGE_ALIGN) u8{[_]?[*]align(PAGE_ALIGN) u8{null} ** DEQUE_SIZE} ** NUM_BUCKETS,
                .tops = [_]usize{DEQUE_SIZE-1} ** NUM_BUCKETS,
                .sizes = [_]usize{0} ** NUM_BUCKETS,
                .size_map = std.AutoHashMap(usize, usize).init(backing_allocator),
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            // Free all cached slabs
            for (0..NUM_BUCKETS) |bucket_idx| {
                while (self.sizes[bucket_idx] > 0) {
                    if (self.popSlab(bucket_idx)) |ptr| {
                        const log2_align = math.log2(PAGE_ALIGN);
                        self.backing_allocator.rawFree(
                            ptr[0..self.sizeFromBucket(bucket_idx)],
                            log2_align,
                            @returnAddress()
                        );
                    }
                }
            }
            self.size_map.deinit();
        }

        pub fn allocator(self: *Self) Allocator {
            return .{
                .ptr = self,
                .vtable = &.{
                    .alloc = alloc,
                    .resize = Allocator.noResize,
                    .free = free,
                },
            };
        }

        // Convert a size to a bucket index
        fn bucketFromSize(self: *Self, size: usize) ?usize {
            _ = self;

            if (size <= THRESHOLD) {
                return null; // Let backing allocator handle small allocations
            }

            // Find the next power of 2 that is >= size
            const bucket = math.log2_int(usize, math.ceilPowerOfTwo(usize, size) catch return null);
            if (bucket >= NUM_BUCKETS) {
                return null; // Too large for our buckets
            }

            return bucket;
        }

        // Convert a bucket index to the size it represents
        fn sizeFromBucket(self: *Self, bucket_idx: usize) usize {
            _ = self;
            return @as(usize, 1) << @as(math.Log2Int(usize), @intCast(bucket_idx));
        }

        // Push a slab onto the top of the deque for a given bucket
        fn pushSlab(self: *Self, bucket_idx: usize, ptr: [*]align(PAGE_ALIGN) u8) void {
            if (self.sizes[bucket_idx] >= DEQUE_SIZE) {
                // Deque is full - remove the oldest entry and replace it
                const bottom_idx = (self.tops[bucket_idx] + 1) % DEQUE_SIZE;
                const old_ptr = self.slabs[bucket_idx][bottom_idx].?;
                const size = self.sizeFromBucket(bucket_idx);
                const log2_align = math.log2(PAGE_ALIGN);
                // std.log.err("I'm calling rawFree because the deque is full!", .{});
                self.backing_allocator.rawFree(
                    old_ptr[0..size],
                    log2_align,
                    @returnAddress()
                );
                self.sizes[bucket_idx] -= 1;
            }
            const index = (self.tops[bucket_idx] + 1) % DEQUE_SIZE;
            self.slabs[bucket_idx][index] = ptr;
            self.tops[bucket_idx] = index;
            self.sizes[bucket_idx] += 1;
        }

        // Pop a slab from the top of the deque for a given bucket
        fn popSlab(self: *Self, bucket_idx: usize) ?[*]align(PAGE_ALIGN) u8 {
            if (self.sizes[bucket_idx] == 0) {
                return null;
            }

            const index = self.tops[bucket_idx] % DEQUE_SIZE;
            self.tops[bucket_idx] = (index + DEQUE_SIZE - 1) % DEQUE_SIZE;
            const ptr = self.slabs[bucket_idx][index];
            self.slabs[bucket_idx][index] = null;
            self.sizes[bucket_idx] -= 1;

            return ptr;
        }

        fn alloc(
            ctx: *anyopaque,
            len: usize,
            log2_ptr_align_u8: u8,
            ret_addr: usize
        ) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const log2_ptr_align = @as(Allocator.Log2Align, @intCast(log2_ptr_align_u8));

            // Calculate the effective alignment - use at least PAGE_ALIGN for large allocations
            const alignment = @max(@as(usize, 1) << @as(math.Log2Int(usize), @intCast(log2_ptr_align)), PAGE_ALIGN);

            // For non-standard alignments, delegate to backing allocator
            if ((len > THRESHOLD and alignment > PAGE_ALIGN)) {
                return self.backing_allocator.rawAlloc(len, log2_ptr_align, ret_addr);
            }

            // For large allocations, try to reuse a cached slab
            if (self.bucketFromSize(len)) |bucket_idx| {
                const size = self.sizeFromBucket(bucket_idx);

                // Check if we have a cached slab
                if (self.popSlab(bucket_idx)) |ptr| {
                    // We found a suitable slab in our cache
                    // Record the original size for later freeing
                    self.size_map.put(@intFromPtr(ptr), size) catch {
                        // If we can't track the size, we can't reuse this allocation
                        const log2_page_align = math.log2(PAGE_ALIGN);
                        self.backing_allocator.rawFree(
                            ptr[0..size],
                            log2_page_align,
                            ret_addr
                        );
                        // Fall through to allocate a new one
                    };

                    return ptr;
                }

                // No cached slab, allocate a new one
                const log2_page_align = math.log2(PAGE_ALIGN);
                // std.log.err("I'm calling rawAlloc with size {d}!", .{size});
                if (self.backing_allocator.rawAlloc(size, log2_page_align, ret_addr)) |ptr| {
                    // Record the original size for later freeing
                    self.size_map.put(@intFromPtr(ptr), size) catch {
                        // If we can't track the size, free it and return null
                        self.backing_allocator.rawFree(
                            ptr[0..size],
                            log2_page_align,
                            ret_addr
                        );
                        return null;
                    };

                    return ptr;
                }

                // Allocation failed
                return null;
            }

            // The size is too large for our buckets, delegate to backing allocator
            return self.backing_allocator.rawAlloc(len, log2_ptr_align, ret_addr);
        }

        fn free(
            ctx: *anyopaque,
            ptr: []u8,
            log2_ptr_align_u8: u8,
            ret_addr: usize
        ) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const log2_ptr_align = @as(Allocator.Log2Align, @intCast(log2_ptr_align_u8));

            // Early return for null allocations
            if (ptr.len == 0) return;

            // Check if this is one of our cached allocations
            if (self.size_map.get(@intFromPtr(ptr.ptr))) |size| {
                // This is one of our allocations

                // Find the bucket for this size
                if (self.bucketFromSize(size)) |bucket_idx| {
                    // Cache this allocation for reuse
                    const aligned_ptr: [*]align(PAGE_ALIGN) u8 = @ptrCast(@alignCast(ptr.ptr));
                    self.pushSlab(bucket_idx, aligned_ptr);

                    // We don't want to remove from size_map as we're keeping the allocation
                    return;
                }

                // If it doesn't fit in our buckets, fall through to regular free
                _ = self.size_map.remove(@intFromPtr(ptr.ptr));
            }

            // For small allocations or those we don't track, delegate to backing allocator
            self.backing_allocator.rawFree(ptr, log2_ptr_align, ret_addr);
        }
    };
}
