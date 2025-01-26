const std = @import("std");

pub fn copyKVCacheChunk(dst: []f16, src: []const f16, dst_offset: usize, src_offset: usize, len: usize) void {
    @memcpy(dst[dst_offset..][0..len], src[src_offset..][0..len]);
}
