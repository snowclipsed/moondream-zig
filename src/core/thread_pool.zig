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
        initialized: bool = false,
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

        // Initialize worker threads but don't start them yet
        for (pool.workers) |*worker| {
            worker.pool = pool;
            worker.initialized = false;
        }

        // Start worker threads with proper error handling
        errdefer {
            // Signal threads to stop
            pool.running.store(false, .seq_cst);
            // Wake up all threads
            pool.mutex.lock();
            pool.condition.broadcast();
            pool.mutex.unlock();
            // Join all initialized threads
            for (pool.workers) |*worker| {
                if (worker.initialized) {
                    worker.thread.join();
                }
            }
        }

        for (pool.workers) |*worker| {
            worker.thread = try Thread.spawn(.{}, workerFn, .{worker});
            worker.initialized = true;
        }

        return pool;
    }

    /// Shut down the thread pool and clean up resources
    pub fn deinit(self: *Self) void {
        // Wait for all tasks to complete first
        self.waitForAll();

        // Signal all threads to exit
        self.running.store(false, .seq_cst);

        // Wake up all threads
        {
            self.mutex.lock();
            self.condition.broadcast();
            self.mutex.unlock();
        }

        // Wait for all threads to finish
        for (self.workers) |worker| {
            if (worker.initialized) {
                worker.thread.join();
            }
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

        // First try adding to queue, then increment active tasks only on success
        try self.task_queue.writeItem(Task{ .func = func, .context = context });
        _ = self.active_tasks.fetchAdd(1, .seq_cst);

        // Wake all threads to ensure tasks are picked up efficiently
        self.condition.broadcast();
    }

    /// Worker thread function
    fn workerFn(worker: *Worker) void {
        const pool = worker.pool;

        while (true) {
            // Check running state first
            if (!pool.running.load(.seq_cst)) {
                return;
            }

            // Try to get a task
            var task: ?Task = null;

            // Critical section for task access
            pool.mutex.lock();

            // Check for tasks
            if (pool.task_queue.readItem()) |t| {
                task = t;
                pool.mutex.unlock();
            } else {
                // Check running state again inside critical section
                if (!pool.running.load(.seq_cst)) {
                    pool.mutex.unlock();
                    return;
                }

                // No tasks and still running, wait for notification
                pool.condition.wait(&pool.mutex);
                pool.mutex.unlock();
                continue;
            }

            // Execute task (outside of critical section)
            if (task) |t| {
                t.func(t.context);

                const prev_count = pool.active_tasks.fetchSub(1, .seq_cst);
                if (prev_count == 1) {
                    // Last task completed, wake up anyone waiting
                    pool.waiting_mutex.lock();
                    pool.waiting_condition.broadcast();
                    pool.waiting_mutex.unlock();
                }
            }
        }
    }

    /// Wait for all tasks to complete
    pub fn waitForAll(self: *Self) void {
        // Fast path - check if no tasks are active
        if (self.active_tasks.load(.seq_cst) == 0) return;

        // Wait for tasks to complete
        self.waiting_mutex.lock();
        defer self.waiting_mutex.unlock();

        while (self.active_tasks.load(.seq_cst) > 0) {
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
    init_mutex.lock();
    defer init_mutex.unlock();

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
