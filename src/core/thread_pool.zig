const std = @import("std");
const Allocator = std.mem.Allocator;
const atomic = std.atomic;
const Thread = std.Thread;
const Mutex = Thread.Mutex;
const Condition = Thread.Condition;

/// Global thread pool for all multithreaded operations
pub const ThreadPool = struct {
    const Self = @This();

    /// Task function type - takes a pointer to arbitrary context data
    pub const TaskFn = *const fn (ctx: *anyopaque) void;

    /// Task representation
    const Task = struct {
        func: TaskFn,
        context: *anyopaque,
    };

    /// Thread worker state
    const Worker = struct {
        thread: Thread,
        pool: *ThreadPool,
    };

    /// Fields
    allocator: Allocator,
    workers: []Worker,
    task_queue: std.fifo.LinearFifo(Task, .Dynamic),
    mutex: Mutex = .{},
    condition: Condition = .{},
    running: atomic.Value(bool),
    active_tasks: atomic.Value(usize),
    waiting_mutex: Mutex = .{},
    waiting_condition: Condition = .{},

    /// Create a thread pool with specified number of threads
    pub fn init(allocator: Allocator, thread_count: usize) !*Self {
        const pool = try allocator.create(Self);
        errdefer allocator.destroy(pool);

        pool.* = Self{
            .allocator = allocator,
            .workers = try allocator.alloc(Worker, thread_count),
            .task_queue = std.fifo.LinearFifo(Task, .Dynamic).init(allocator),
            .running = atomic.Value(bool).init(true),
            .active_tasks = atomic.Value(usize).init(0),
        };
        errdefer pool.task_queue.deinit();
        errdefer allocator.free(pool.workers);

        // Initialize and start worker threads
        for (pool.workers) |*worker| {
            worker.pool = pool;
            worker.thread = try Thread.spawn(.{}, workerFn, .{worker});
        }

        return pool;
    }

    /// Shut down the thread pool and clean up resources
    pub fn deinit(self: *Self) void {
        // Signal all threads to exit
        self.running.store(false, .release);

        // Wake up all threads
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.condition.broadcast();
        }

        // Wait for all threads to finish
        for (self.workers) |worker| {
            worker.thread.join();
        }

        // Free resources
        self.allocator.free(self.workers);
        self.task_queue.deinit();
        self.allocator.destroy(self);
    }

    /// Submit a task to the thread pool
    pub fn submit(self: *Self, func: TaskFn, context: *anyopaque) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.task_queue.writeItem(Task{ .func = func, .context = context });
        _ = self.active_tasks.fetchAdd(1, .monotonic);
        self.condition.signal();
    }

    /// Worker thread function
    fn workerFn(worker: *Worker) void {
        const pool = worker.pool;

        while (pool.running.load(.acquire)) {
            // Try to get a task
            var task: ?Task = null;

            // Critical section: check for and potentially get a task
            pool.mutex.lock();
            while (pool.task_queue.readItem()) |t| {
                task = t;
                break;
            } else if (pool.running.load(.acquire)) {
                // Wait for new tasks
                pool.condition.wait(&pool.mutex);
                pool.mutex.unlock();
                continue;
            }
            pool.mutex.unlock();

            // If we have a task, execute it
            if (task) |t| {
                t.func(t.context);
                const prev_count = pool.active_tasks.fetchSub(1, .release);
                if (prev_count == 1) {
                    // Last task completed, wake up anyone waiting
                    pool.waiting_mutex.lock();
                    defer pool.waiting_mutex.unlock();
                    pool.waiting_condition.broadcast();
                }
            } else {
                // No tasks and not running
                break;
            }
        }
    }

    /// Wait for all tasks to complete
    pub fn waitForAll(self: *Self) void {
        // Fast path - check if no tasks are active
        if (self.active_tasks.load(.acquire) == 0) return;

        // Wait for tasks to complete
        self.waiting_mutex.lock();
        defer self.waiting_mutex.unlock();

        while (self.active_tasks.load(.acquire) > 0) {
            self.waiting_condition.wait(&self.waiting_mutex);
        }
    }
};

// Global instance and management functions
pub var global_instance: ?*ThreadPool = null;
var init_mutex: Mutex = .{};

/// Initialize the global thread pool (optimal thread count)
pub fn init(allocator: Allocator) !void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (global_instance != null) return;

    const thread_count = try Thread.getCpuCount();
    global_instance = try ThreadPool.init(allocator, thread_count);
}

/// Initialize with a specific thread count
pub fn initWithThreadCount(allocator: Allocator, thread_count: usize) !void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (global_instance != null) return;

    global_instance = try ThreadPool.init(allocator, thread_count);
}

/// Shut down the global thread pool
pub fn deinit() void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (global_instance) |pool| {
        pool.deinit();
        global_instance = null;
    }
}

/// Get the global thread pool instance
pub fn getInstance() !*ThreadPool {
    if (global_instance) |pool| {
        return pool;
    }
    return error.ThreadPoolNotInitialized;
}

/// Helper to submit a typed task
pub fn submitTask(comptime Context: type, func: fn (ctx: *Context) void, context: *Context) !void {
    const pool = try getInstance();

    const GenericFn = struct {
        fn wrapper(ctx: *anyopaque) void {
            func(@ptrCast(@alignCast(ctx)));
        }
    };

    try pool.submit(&GenericFn.wrapper, context);
}

/// Convenience function to wait for all tasks to complete
pub fn waitForAll() !void {
    const pool = try getInstance();
    pool.waitForAll();
}
