#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DXC=""
for c in "/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe" \
         "/c/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/dxc.exe" \
         /c/VulkanSDK/*/Bin/dxc.exe; do
    [ -x "$c" ] && DXC="$c" && break
done
if [ -z "$DXC" ]; then echo "*** dxc.exe not found ***"; exit 1; fi
EXE="$ROOT/build/Release/nori-dxr.exe"
WHICH="${1:-all}"

echo "[compile] dxc -T lib_6_5 Shaders.hlsl (RayQuery default)"
if ! "$DXC" -T lib_6_5 -Fo "$ROOT/build/Release/Shaders.cso" "$ROOT/shaders/Shaders.hlsl"; then
    echo "*** COMPILE FAILED ***"; exit 1
fi

rc=0
check() {
    local name="$1" scene="$2" dir="$3" want="$4" snap="${5:-snapshot_64.exr}"
    [ "$WHICH" != "all" ] && [ "$WHICH" != "$name" ] && return 0
    "$EXE" "$scene" --headless >/dev/null 2>&1
    local got
    got=$(sha256sum "$dir/$snap" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
        echo "PASS  $name"
    else
        echo "FAIL  $name  got=${got:0:12}  want=${want:0:12}"; rc=1
    fi
}


check cbox   "$ROOT/scenes/final_scenes/cbox/cbox_verify.xml"   "$ROOT/scenes/final_scenes/cbox" 40a872ac57c3824aadc422a4de410259449d67f85bb9dacda7e62b98bd82d6d2 snapshot_2048.exr
check mats   "$ROOT/scenes/final_scenes/cbox/mats_verify.xml"   "$ROOT/scenes/final_scenes/cbox" b4cd0d956ae5971aaeee8dfe32ba43243ad77fdf8410fa443d5929bc5289d20a
check hair   "$ROOT/scenes/final_scenes/hair/hair_verify.xml"   "$ROOT/scenes/final_scenes/hair" 964a807b0271433012e7f13c6e9d7b7c863a19aea9b520589f345be1607bfb37
check hetero "$ROOT/scenes/final_scenes/cbox/hetero_verify.xml" "$ROOT/scenes/final_scenes/cbox" 6a39d7e1d0ad8b771b5f774d4d460727c6f9b658304eb905841ea5fa0ad69629

[ $rc -eq 0 ] && echo "ALL GREEN" || echo "*** MISMATCH ***"
exit $rc
