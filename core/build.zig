const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const exe = b.addExecutable(.{
        .name = "moondream",
        .root_source_file = b.path("src/moondream.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.addIncludePath(b.path("./dependencies/ggml/include/"));
    exe.addIncludePath(b.path("./dependencies/ggml/include/ggml"));
    exe.addCSourceFiles(.{
        .files = &.{
            "./dependencies/ggml/src/ggml.c",
            "./dependencies/ggml/src/ggml-alloc.c",
            "./dependencies/ggml/src/ggml-backend.c",
            "./dependencies/ggml/src/ggml-quants.c",
        },
        .flags = &.{
            "-std=c11",
            "-D_GNU_SOURCE",
            "-D_XOPEN_SOURCE=600",
        },
    });
    exe.linkLibC();
    // exe.linkLibCpp();
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "run moondream");
    run_step.dependOn(&run_cmd.step);
}
