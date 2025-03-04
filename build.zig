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
        .file = .{ .cwd_relative = "src/dependencies/stb_image.c" },
        .flags = &[_][]const u8{
            "-Wall",
            "-Wextra",
            "-O3",
            "-DSTB_IMAGE_IMPLEMENTATION",
        },
    });

    stb_resize.addCSourceFile(.{
        .file = .{ .cwd_relative = "src/dependencies/stb_image_resize2.c" },
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

    // ---- Chat Client Executable ----
    const chat_exe = b.addExecutable(.{
        .name = "moonchat",
        .root_source_file = .{ .cwd_relative = "src/moonchat.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add dependencies folder for headers
    chat_exe.addIncludePath(.{ .cwd_relative = "src/dependencies" });

    // Link with our static libraries
    chat_exe.linkLibrary(stb_image);
    chat_exe.linkLibrary(stb_resize);
    chat_exe.linkLibC();

    b.installArtifact(chat_exe);

    // ---- Single Turn Client Executable ----
    const api_exe = b.addExecutable(.{
        .name = "moondream",
        .root_source_file = .{ .cwd_relative = "src/moondream.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add dependencies folder for headers
    api_exe.addIncludePath(.{ .cwd_relative = "src/dependencies" });

    // Link with our static libraries
    api_exe.linkLibrary(stb_image);
    api_exe.linkLibrary(stb_resize);
    api_exe.linkLibC();

    b.installArtifact(api_exe);

    // Install the static libraries if needed
    b.installArtifact(stb_image);
    b.installArtifact(stb_resize);

    // Create specific build steps for each executable
    const build_chat_step = b.step("chat", "Build the chat client");
    build_chat_step.dependOn(&b.addInstallArtifact(chat_exe, .{}).step);

    const build_api_step = b.step("api", "Build the single turn client");
    build_api_step.dependOn(&b.addInstallArtifact(api_exe, .{}).step);

    // Create run steps for each executable
    const run_chat_cmd = b.addRunArtifact(chat_exe);
    if (b.args) |args| run_chat_cmd.addArgs(args);
    const run_chat_step = b.step("run-chat", "Run the chat client");
    run_chat_step.dependOn(&run_chat_cmd.step);

    const run_api_cmd = b.addRunArtifact(api_exe);
    if (b.args) |args| run_api_cmd.addArgs(args);
    const run_api_step = b.step("run-api", "Run the single turn client");
    run_api_step.dependOn(&run_api_cmd.step);

    // Default run step uses the chat client for backward compatibility
    const run_cmd = b.addRunArtifact(chat_exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the chat client (default)");
    run_step.dependOn(&run_cmd.step);

    // Tests configuration
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
            .root_source_file = .{ .cwd_relative = "src/core/ops_test.zig" },
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
}
