/**
 * Decode an IEEE 754 binary16 (half-precision float) bit pattern into a JS number.
 */
export function float16ToFloat32(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x03ff;

  if (exp === 0) {
    if (frac === 0) return sign * 0;
    return sign * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 31) {
    return frac === 0 ? sign * Infinity : NaN;
  }
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/**
 * Decode a little-endian float16 slice from an ArrayBuffer into a Float32Array.
 */
export function decodeFloat16Buffer(
  buffer: ArrayBuffer,
  byteOffset: number,
  byteLength: number
): Float32Array {
  const count = byteLength / 2;
  const view = new DataView(buffer, byteOffset, byteLength);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = float16ToFloat32(view.getUint16(i * 2, true));
  }
  return out;
}
