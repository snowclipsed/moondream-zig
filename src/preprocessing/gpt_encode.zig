const std = @import("std");
const io = std.io;

const ENCODE_TABLE: [256][]const u8 = .{
    "\xc4\x80",
    "\xc4\x81",
    "\xc4\x82",
    "\xc4\x83",
    "\xc4\x84",
    "\xc4\x85",
    "\xc4\x86",
    "\xc4\x87",
    "\xc4\x88",
    "\xc4\x89",
    "\xc4\x8a",
    "\xc4\x8b",
    "\xc4\x8c",
    "\xc4\x8d",
    "\xc4\x8e",
    "\xc4\x8f",
    "\xc4\x90",
    "\xc4\x91",
    "\xc4\x92",
    "\xc4\x93",
    "\xc4\x94",
    "\xc4\x95",
    "\xc4\x96",
    "\xc4\x97",
    "\xc4\x98",
    "\xc4\x99",
    "\xc4\x9a",
    "\xc4\x9b",
    "\xc4\x9c",
    "\xc4\x9d",
    "\xc4\x9e",
    "\xc4\x9f",
    "\xc4\xa0",
    "\x21",
    "\x22",
    "\x23",
    "\x24",
    "\x25",
    "\x26",
    "\x27",
    "\x28",
    "\x29",
    "\x2a",
    "\x2b",
    "\x2c",
    "\x2d",
    "\x2e",
    "\x2f",
    "\x30",
    "\x31",
    "\x32",
    "\x33",
    "\x34",
    "\x35",
    "\x36",
    "\x37",
    "\x38",
    "\x39",
    "\x3a",
    "\x3b",
    "\x3c",
    "\x3d",
    "\x3e",
    "\x3f",
    "\x40",
    "\x41",
    "\x42",
    "\x43",
    "\x44",
    "\x45",
    "\x46",
    "\x47",
    "\x48",
    "\x49",
    "\x4a",
    "\x4b",
    "\x4c",
    "\x4d",
    "\x4e",
    "\x4f",
    "\x50",
    "\x51",
    "\x52",
    "\x53",
    "\x54",
    "\x55",
    "\x56",
    "\x57",
    "\x58",
    "\x59",
    "\x5a",
    "\x5b",
    "\x5c",
    "\x5d",
    "\x5e",
    "\x5f",
    "\x60",
    "\x61",
    "\x62",
    "\x63",
    "\x64",
    "\x65",
    "\x66",
    "\x67",
    "\x68",
    "\x69",
    "\x6a",
    "\x6b",
    "\x6c",
    "\x6d",
    "\x6e",
    "\x6f",
    "\x70",
    "\x71",
    "\x72",
    "\x73",
    "\x74",
    "\x75",
    "\x76",
    "\x77",
    "\x78",
    "\x79",
    "\x7a",
    "\x7b",
    "\x7c",
    "\x7d",
    "\x7e",
    "\xc4\xa1",
    "\xc4\xa2",
    "\xc4\xa3",
    "\xc4\xa4",
    "\xc4\xa5",
    "\xc4\xa6",
    "\xc4\xa7",
    "\xc4\xa8",
    "\xc4\xa9",
    "\xc4\xaa",
    "\xc4\xab",
    "\xc4\xac",
    "\xc4\xad",
    "\xc4\xae",
    "\xc4\xaf",
    "\xc4\xb0",
    "\xc4\xb1",
    "\xc4\xb2",
    "\xc4\xb3",
    "\xc4\xb4",
    "\xc4\xb5",
    "\xc4\xb6",
    "\xc4\xb7",
    "\xc4\xb8",
    "\xc4\xb9",
    "\xc4\xba",
    "\xc4\xbb",
    "\xc4\xbc",
    "\xc4\xbd",
    "\xc4\xbe",
    "\xc4\xbf",
    "\xc5\x80",
    "\xc5\x81",
    "\xc5\x82",
    "\xc2\xa1",
    "\xc2\xa2",
    "\xc2\xa3",
    "\xc2\xa4",
    "\xc2\xa5",
    "\xc2\xa6",
    "\xc2\xa7",
    "\xc2\xa8",
    "\xc2\xa9",
    "\xc2\xaa",
    "\xc2\xab",
    "\xc2\xac",
    "\xc5\x83",
    "\xc2\xae",
    "\xc2\xaf",
    "\xc2\xb0",
    "\xc2\xb1",
    "\xc2\xb2",
    "\xc2\xb3",
    "\xc2\xb4",
    "\xc2\xb5",
    "\xc2\xb6",
    "\xc2\xb7",
    "\xc2\xb8",
    "\xc2\xb9",
    "\xc2\xba",
    "\xc2\xbb",
    "\xc2\xbc",
    "\xc2\xbd",
    "\xc2\xbe",
    "\xc2\xbf",
    "\xc3\x80",
    "\xc3\x81",
    "\xc3\x82",
    "\xc3\x83",
    "\xc3\x84",
    "\xc3\x85",
    "\xc3\x86",
    "\xc3\x87",
    "\xc3\x88",
    "\xc3\x89",
    "\xc3\x8a",
    "\xc3\x8b",
    "\xc3\x8c",
    "\xc3\x8d",
    "\xc3\x8e",
    "\xc3\x8f",
    "\xc3\x90",
    "\xc3\x91",
    "\xc3\x92",
    "\xc3\x93",
    "\xc3\x94",
    "\xc3\x95",
    "\xc3\x96",
    "\xc3\x97",
    "\xc3\x98",
    "\xc3\x99",
    "\xc3\x9a",
    "\xc3\x9b",
    "\xc3\x9c",
    "\xc3\x9d",
    "\xc3\x9e",
    "\xc3\x9f",
    "\xc3\xa0",
    "\xc3\xa1",
    "\xc3\xa2",
    "\xc3\xa3",
    "\xc3\xa4",
    "\xc3\xa5",
    "\xc3\xa6",
    "\xc3\xa7",
    "\xc3\xa8",
    "\xc3\xa9",
    "\xc3\xaa",
    "\xc3\xab",
    "\xc3\xac",
    "\xc3\xad",
    "\xc3\xae",
    "\xc3\xaf",
    "\xc3\xb0",
    "\xc3\xb1",
    "\xc3\xb2",
    "\xc3\xb3",
    "\xc3\xb4",
    "\xc3\xb5",
    "\xc3\xb6",
    "\xc3\xb7",
    "\xc3\xb8",
    "\xc3\xb9",
    "\xc3\xba",
    "\xc3\xbb",
    "\xc3\xbc",
    "\xc3\xbd",
    "\xc3\xbe",
    "\xc3\xbf",
};

pub fn encode(out: []u8, in: []const u8) ![]u8 {
    var i: usize = 0;
    var out_idx: usize = 0;
    while (i < in.len) : (i += 1) {
        const slice = ENCODE_TABLE[in[i]];
        const end_idx = out_idx + slice.len;
        if (end_idx > out.len) {
            return error.YourSliceIsTooSmall;
        }
        @memcpy(out[out_idx..end_idx], slice);
        out_idx = end_idx;
    }
    return out[0..out_idx];
}

pub fn get_encoded_len(in: []const u8) usize {
    var i: usize = 0;
    var out_idx: usize = 0;
    while (i < in.len) : (i += 1) {
        const slice = ENCODE_TABLE[in[i]];
        out_idx += slice.len;
    }
    return out_idx;
}

fn make_decode_table() [0x10000]u8 {
    @setEvalBranchQuota(100000);
    var ret: [0x10000]u8 = undefined;
    @memset(&ret, 0xaa);
    for (ENCODE_TABLE, 0..) |slice, i| {
        if (slice.len == 1) {
            var n: usize = 0;
            while (n < 256) : (n += 1) {
                const idx: u16 = @bitCast([2]u8{slice[0], n});
                ret[idx] = i;
            }
        }
        if (slice.len == 2) {
            const value: u16 = @bitCast((slice.ptr)[0..2].*);
            ret[value] = @as(u8, @intCast(i));
        }
    }
    return ret;
}

const DECODE_TABLE = make_decode_table();

pub fn decode(out: []u8, in: []const u8) ![]u8 {
    var i: usize = 0;
    var out_idx: usize = 0;
    while (i < in.len) : (i += 1) {
        var value: u16 = 0;
        if (i+1 < in.len) {
            value = @bitCast((in.ptr + i)[0..2].*);
        } else {
            value = in[i];
        }
        const out_byte = DECODE_TABLE[value];
        if (out_idx >= out.len) {
            return error.YourSliceIsTooSmall;
        }
        out[out_idx] = out_byte;
        out_idx += 1;
        i += (value >> 7) & 1;
    }
    return out[0..out_idx];
}

pub fn get_decoded_len(in: []const u8) usize {
    var i: usize = 0;
    var out_idx: usize = 0;
    while (i < in.len) : (i += 1) {
        var value: u16 = 0;
        if (i+1 < in.len) {
            value = @bitCast((in.ptr + i)[0..2].*);
        } else {
            value = in[i];
        }
        i += (value >> 7) & 1;
        out_idx += 1;
    }
    return out_idx;
}
