const std = @import("std");

pub fn build(b: *std.Build) void {
    // const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const exe = b.addExecutable(.{
        .name = "matmul",
        .root_source_file = b.path("matmul.zig"),
        .target = b.host,
        .optimize = optimize,
    });

    exe.linkSystemLibrary("openblas");
    exe.linkLibC();

    b.installArtifact(exe);
    const test_targets = [_]std.Target.Query{
        .{}, // native
        // .{
        //     .cpu_arch = .x86_64,
        //     .os_tag = .linux,
        // },
        // .{
        //     .cpu_arch = .aarch64,
        //     .os_tag = .macos,
        // },
    };
    const test_step = b.step("test", "Run unit tests");

    for (test_targets) |target| {
        const unit_tests = b.addTest(.{
            .root_source_file = b.path("matmul.zig"),
            .target = b.resolveTargetQuery(target),
        });

        unit_tests.linkSystemLibrary("openblas");
        unit_tests.linkLibC();

        const run_unit_tests = b.addRunArtifact(unit_tests);
        test_step.dependOn(&run_unit_tests.step);
    }

    const run_exe = b.addRunArtifact(exe);

    const run_step = b.step("run", "Run Matmul.");
    run_step.dependOn(&run_exe.step);
}
