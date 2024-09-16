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

    // Start of adding image libraries //

    // Add dependencies folder as library path to search for your libstb_image.so and libstb_resize2.c file

    exe.addLibraryPath(std.Build.LazyPath{ .src_path = .{
        .owner = b,
        .sub_path = "dependencies",
    } });

    // Link the stb_image library
    exe.linkSystemLibrary("stb_image");
    exe.linkSystemLibrary("stb_image_resize2");
    exe.linkLibC();

    // End of adding image libraries //

    // Linking GGML //

    // exe.addIncludePath(b.path("./dependencies/ggml/include/"));
    // exe.addIncludePath(b.path("./dependencies/ggml/include/ggml"));
    // exe.addCSourceFiles(.{
    //     .files = &.{
    //         "./dependencies/ggml/src/ggml.c",
    //         "./dependencies/ggml/src/ggml-alloc.c",
    //         "./dependencies/ggml/src/ggml-backend.c",
    //         "./dependencies/ggml/src/ggml-quants.c",
    //     },
    //     .flags = &.{
    //         "-std=c11",
    //         "-D_GNU_SOURCE",
    //         "-D_XOPEN_SOURCE=600",
    //     },
    // });
    // exe.linkLibC();
    // exe.linkLibCpp();

    // End of linking GGML //

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

    for (test_targets) |test_target| {
        const unit_tests = b.addTest(.{
            .root_source_file = b.path("src/moondream.zig"),
            .target = b.resolveTargetQuery(test_target),
        });

        unit_tests.linkSystemLibrary("openblas");
        unit_tests.linkLibC();

        const run_unit_tests = b.addRunArtifact(unit_tests);
        test_step.dependOn(&run_unit_tests.step);
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "run moondream");
    run_step.dependOn(&run_cmd.step);
}
