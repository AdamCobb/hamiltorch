"""Python port of the dataviz palette validator (same thresholds/math)."""
import math, sys
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
MACHADO = {
 "protan": [[0.152286,1.052583,-0.204868],[0.114503,0.786281,0.099216],[-0.003882,-0.048116,1.051998]],
 "deutan": [[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],[-0.011820,0.042940,0.968881]],
}
def hex2srgb(h):
    h = h.strip().lstrip("#"); return [int(h[i:i+2],16)/255 for i in (0,2,4)]
def s2lin(c): return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
def lin(h): return [s2lin(c) for c in hex2srgb(h)]
def relLum(h):
    r,g,b = lin(h); return 0.2126*r+0.7152*g+0.0722*b
def contrast(a,b):
    hi,lo = sorted([relLum(a),relLum(b)], reverse=True); return (hi+0.05)/(lo+0.05)
def oklab_from_lin(rgb):
    r,g,b = rgb
    l = (0.4122214708*r+0.5363325363*g+0.0514459929*b)**(1/3)
    m = (0.2119034982*r+0.6806995451*g+0.1073969566*b)**(1/3)
    s = (0.0883024619*r+0.2817188376*g+0.6299787005*b)**(1/3)
    return [0.2104542553*l+0.7936177850*m-0.0040720468*s,
            1.9779984951*l-2.4285922050*m+0.4505937099*s,
            0.0259040371*l+0.7827717662*m-0.8086757660*s]
def oklch(h):
    L,a,b = oklab_from_lin(lin(h)); return L, math.hypot(a,b)
def simulate(h, kind):
    r,g,b = lin(h); M = MACHADO[kind]
    return [min(1,max(0, M[i][0]*r+M[i][1]*g+M[i][2]*b)) for i in range(3)]
def deltaE(h1,h2,kind=None):
    a = oklab_from_lin(simulate(h1,kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2,kind) if kind else lin(h2))
    return 100*math.dist(a,b)

def validate(pal, mode="light", pairs="adjacent"):
    surf = SURFACE[mode]; lo,hi = BAND[mode]; ok = True
    off = [(c, round(oklch(c)[0],3)) for c in pal if not (lo <= oklch(c)[0] <= hi)]
    print(f"  1 lightness band [{lo},{hi}]: {'PASS' if not off else 'FAIL '+str(off)}"); ok &= not off
    lowc = [(c, round(oklch(c)[1],3)) for c in pal if oklch(c)[1] < CHROMA_FLOOR]
    print(f"  2 chroma floor {CHROMA_FLOOR}: {'PASS' if not lowc else 'FAIL '+str(lowc)}"); ok &= not lowc
    idx = [(i,i+1) for i in range(len(pal)-1)] if pairs=="adjacent" else \
          [(i,j) for i in range(len(pal)) for j in range(i+1,len(pal))]
    worst = min(((min(deltaE(pal[i],pal[j],"protan"), deltaE(pal[i],pal[j],"deutan")), i, j) for i,j in idx))
    st = "PASS" if worst[0] >= CVD_TARGET else ("FLOOR" if worst[0] >= CVD_FLOOR else "FAIL")
    print(f"  3 CVD sep ({pairs}) min dE={worst[0]:.1f} at slots {worst[1]+1},{worst[2]+1}: {st}")
    ok &= worst[0] >= CVD_FLOOR
    wn = min(((deltaE(pal[i],pal[j]), i, j) for i,j in idx))
    print(f"  4 normal-vision floor {NORMAL_FLOOR} min dE={wn[0]:.1f}: {'PASS' if wn[0]>=NORMAL_FLOOR else 'FAIL'}")
    ok &= wn[0] >= NORMAL_FLOOR
    bad = [(c, round(contrast(c,surf),2)) for c in pal if contrast(c,surf) < CONTRAST_MIN]
    print(f"  5 contrast vs {surf} >= {CONTRAST_MIN}: {'PASS' if not bad else 'WARN '+str(bad)}")
    return ok

if __name__ == "__main__":
    pal = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    mode = sys.argv[2] if len(sys.argv) > 2 else "light"
    pairs = sys.argv[3] if len(sys.argv) > 3 else "adjacent"
    print(f"palette {pal} mode={mode} pairs={pairs}")
    print("RESULT:", "PASS" if validate(pal, mode, pairs) else "FAIL")
