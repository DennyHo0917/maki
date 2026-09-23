-- Rust parity for provider plugins that port a bespoke Rust parser. The Rust
-- side reads JSON with serde_json and prints with `format!`, and these helpers
-- reproduce its numbers bit for bit. Other plugins are better off with
-- `maki.json`.
--
-- Luau has a single number type, so `maki.json.decode` gives `8192` and
-- `8192.0` the same value, while serde_json's `as_u64` accepts only the first.
-- `M.decode` remembers which numbers were floats in the source text, and the
-- readers take the container and key (`M.as_u32(m, "context_length")` mirrors
-- `m["context_length"].as_u64().and_then(|v| u32::try_from(v).ok())`). A
-- missing key, a JSON null, a non-table container or the wrong type reads as
-- nil. Tables that did not come from `M.decode` carry no float marks, so there
-- a whole-valued float passes as an integer.
--
-- Luau numbers are doubles: a u64 above 2^53 comes back rounded, and
-- u64::MAX reads as 2^64.
local M = {}

local U32_MAX = 4294967295
local U64_MAX = 2 ^ 64
local U64_MAX_DIGITS = "18446744073709551615"
local QUOTE = string.byte('"')
local BACKSLASH = string.byte("\\")
local FLOAT_TAG = "\0f64:"
local FLOAT_TAG_JSON = '"\\u0000f64:'
local RESERVED_STRING = "string starts with the reserved float tag"
local NAN = 0 / 0

local float_keys = setmetatable({}, { __mode = "k" })

local function string_end(text, start)
  local pos = start + 1
  while true do
    local found = string.find(text, '["\\]', pos)
    if not found then
      return #text
    end
    if string.byte(text, found) == QUOTE then
      return found
    end
    pos = found + 2
  end
end

-- serde_json reads a bare integer past u64::MAX as a float.
local function overflows_u64(lexeme)
  return #lexeme > #U64_MAX_DIGITS or (#lexeme == #U64_MAX_DIGITS and lexeme > U64_MAX_DIGITS)
end

-- Every number serde_json parses as a float becomes a tagged string holding
-- its index in the returned lexeme list.
local function tag_floats(text)
  local pieces, lexemes = {}, {}
  local pos, copied = 1, 1
  while true do
    local start = string.find(text, '[%-%d"]', pos)
    if not start then
      break
    end
    if string.byte(text, start) == QUOTE then
      if
        string.byte(text, start + 1) == BACKSLASH
        and string.sub(text, start, start + #FLOAT_TAG_JSON - 1) == FLOAT_TAG_JSON
      then
        return nil, nil, RESERVED_STRING
      end
      pos = string_end(text, start) + 1
    else
      local _, int_end = string.find(text, "^%-?%d+", start)
      local stop = int_end or start
      if int_end then
        local _, frac_end = string.find(text, "^%.%d+", stop + 1)
        stop = frac_end or stop
        local _, exp_end = string.find(text, "^[eE][%+%-]?%d+", stop + 1)
        stop = exp_end or stop
        local lexeme = string.sub(text, start, stop)
        if stop > int_end or overflows_u64(lexeme) then
          table.insert(lexemes, lexeme)
          table.insert(pieces, string.sub(text, copied, start - 1))
          table.insert(pieces, FLOAT_TAG_JSON .. #lexemes .. '"')
          copied = stop + 1
        end
      end
      pos = stop + 1
    end
  end
  table.insert(pieces, string.sub(text, copied))
  return table.concat(pieces), lexemes
end

local function restore_floats(node, values)
  for key, value in pairs(node) do
    if type(value) == "table" then
      restore_floats(value, values)
    elseif type(value) == "string" and string.sub(value, 1, #FLOAT_TAG) == FLOAT_TAG then
      node[key] = values[tonumber(string.sub(value, #FLOAT_TAG + 1))]
      float_keys[node] = float_keys[node] or {}
      float_keys[node][key] = true
    end
  end
end

--- `maki.json.decode`, plus a record of which numbers serde_json would read
--- as floats. Returns the value, or nil and an error.
function M.decode(text)
  local tagged, lexemes, err = tag_floats(text)
  if err then
    return nil, err
  end
  if #lexemes == 0 then
    return maki.json.decode(text)
  end
  local wrapper = maki.json.decode("[" .. tagged .. "]")
  if not wrapper then
    return maki.json.decode(text)
  end
  -- serde_json parses each float, so the value matches Rust's to the bit.
  local values = maki.json.decode("[" .. table.concat(lexemes, ",") .. "]")
  restore_floats(wrapper, values)
  return wrapper[1]
end

local function whole_at(tbl, key, max)
  if type(tbl) ~= "table" then
    return nil
  end
  local value = tbl[key]
  if type(value) ~= "number" or value ~= math.floor(value) or value < 0 or value > max or 1 / value < 0 then
    return nil
  end
  local floats = float_keys[tbl]
  if floats and floats[key] then
    return nil
  end
  return value
end

--- serde_json `Value::as_u64` on `tbl[key]`: a non-negative integer, never a
--- float such as `1.0`, `1e3` or `-0`.
function M.as_u64(tbl, key)
  return whole_at(tbl, key, U64_MAX)
end

--- `as_u64` then `u32::try_from(v).ok()`.
function M.as_u32(tbl, key)
  return whole_at(tbl, key, U32_MAX)
end

--- serde_json `Value::as_f64` on `tbl[key]`: any number, integer or float.
function M.as_f64(tbl, key)
  if type(tbl) ~= "table" or type(tbl[key]) ~= "number" then
    return nil
  end
  return tbl[key]
end

--- Rust `s.parse::<f64>().ok()`: no whitespace, no hex, an optional sign,
--- and `inf`, `infinity` or `nan` in any case. nil for a non-string.
function M.parse_f64(s)
  if type(s) ~= "string" then
    return nil
  end
  local sign, body = string.match(s, "^([%+%-]?)(.*)$")
  local word = string.lower(body)
  if word == "inf" or word == "infinity" then
    return sign == "-" and -math.huge or math.huge
  end
  if word == "nan" then
    return NAN
  end
  local mantissa, exponent = string.match(body, "^([%d%.]*)(.*)$")
  local mantissa_ok = string.find(mantissa, "^%d+%.?%d*$") or string.find(mantissa, "^%.%d+$")
  local exponent_ok = exponent == "" or string.find(exponent, "^[eE][%+%-]?%d+$")
  if not (mantissa_ok and exponent_ok) then
    return nil
  end
  return tonumber(s)
end

--- Rust `x as u64`: truncates, NaN and negatives give 0, saturates at the top.
function M.cast_u64(x)
  if x ~= x or x <= 0 then
    return 0
  end
  return math.min(math.floor(x), U64_MAX)
end

--- Rust `x as u32`: truncates, NaN and negatives give 0, saturates at the top.
function M.cast_u32(x)
  return math.min(M.cast_u64(x), U32_MAX)
end

--- Rust `f64::round`: halves round away from zero.
M.round = math.round

--- Rust `format!("{:.n$}", x)`: the exact binary value rounded, ties to even,
--- so `0.125` prints `0.12`. A whole number with `n = 0` prints like Rust's
--- integer `{}`.
function M.fixed(x, n)
  if x ~= x then
    return "NaN"
  end
  if x == math.huge then
    return "inf"
  end
  if x == -math.huge then
    return "-inf"
  end
  return string.format("%." .. n .. "f", x)
end

--- Rust `sort_by`, in place: stable, so elements that are not `less` than
--- each other keep their order. `less(a, b)` is true when `a` sorts first.
function M.stable_sort_by(list, less)
  local len = #list
  local src, dst = list, {}
  local width = 1
  while width < len do
    for lo = 1, len, 2 * width do
      local mid = math.min(lo + width, len + 1)
      local hi = math.min(lo + 2 * width, len + 1)
      local left, right = lo, mid
      for out = lo, hi - 1 do
        if right < hi and (left >= mid or less(src[right], src[left])) then
          dst[out] = src[right]
          right = right + 1
        else
          dst[out] = src[left]
          left = left + 1
        end
      end
    end
    src, dst = dst, src
    width = width * 2
  end
  if src ~= list then
    table.move(src, 1, len, 1, list)
  end
end

return M
