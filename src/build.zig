const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // First create the static libraries for STB
    const stb_image = b.addStaticLibrary(.{
        .name = "stb_image",
        .target = target,
        .optimize = optimize,
    });

    const stb_resize = b.addStaticLibrary(.{
        .name = "stb_image_resize2",
        .target = target,
        .optimize = optimize,
    });

    // Add C source files
    stb_image.addCSourceFile(.{
        .file = .{ .cwd_relative = "dependencies/stb_image.c" },
        .flags = &[_][]const u8{
            "-Wall",
            "-Wextra",
            "-O3",
            "-DSTB_IMAGE_IMPLEMENTATION",
        },
    });

    stb_resize.addCSourceFile(.{
        .file = .{ .cwd_relative = "dependencies/stb_image_resize2.c" },
        .flags = &[_][]const u8{
            "-Wall",
            "-Wextra",
            "-O3",
            "-DSTB_IMAGE_RESIZE_IMPLEMENTATION",
        },
    });

    // Link with C standard library
    stb_image.linkLibC();
    stb_resize.linkLibC();

    // Create the main executable
    const exe = b.addExecutable(.{
        .name = "moondream",
        .root_source_file = .{ .cwd_relative = "moonchat.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add dependencies folder for headers
    exe.addIncludePath(.{ .cwd_relative = "dependencies" });

    // Link with our static libraries
    exe.linkLibrary(stb_image);
    exe.linkLibrary(stb_resize);
    exe.linkLibC();

    b.installArtifact(exe);
    // Also install the static libraries if needed
    b.installArtifact(stb_image);
    b.installArtifact(stb_resize);

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
        const ops_tests = b.addTest(.{
            .root_source_file = .{ .cwd_relative = "ops_test.zig" },
            .target = b.resolveTargetQuery(test_target),
        });

        // Link with our static libraries for tests too
        ops_tests.linkLibrary(stb_image);
        ops_tests.linkLibrary(stb_resize);
        ops_tests.linkSystemLibrary("openblas");
        ops_tests.linkLibC();

        const run_tests = b.addRunArtifact(ops_tests);
        test_step.dependOn(&run_tests.step);
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "run the program");
    run_step.dependOn(&run_cmd.step);
}
