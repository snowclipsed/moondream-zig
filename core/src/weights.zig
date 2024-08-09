const std = @import("std");

pub fn main() void {
    const data: []u8 = &[_]u8{ 0x3C, 0x00, 0x3C, 0x00 }; // Example u8 slice (representing two f16 numbers)

    // Cast the u8 slice to an f16 slice
    const f16_slice: []f16 = @alignCast(@ptrCast(data));

    // Use the f16_slice
    for (f16_slice) |val| {
        std.debug.print("f16 value: {}\n", .{val});
    }
}
