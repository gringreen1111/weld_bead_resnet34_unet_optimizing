"""
DR + 합성 데이터 생성 (비스듬히 카메라 v5 - 완전 async 수정 버전)
======================================
- 시편 위치 고정 (비드와 분리되어 있으므로)
- 조명 inputs:intensity 방식으로 수정
- 모든 DR을 async 루프에서 매 프레임 처리
"""
import random
import asyncio
import carb
import time
import numpy as np
from isaacsim.core.utils.semantics import add_update_semantics
import omni.usd
import omni.replicator.core as rep
from pxr import UsdLux, UsdGeom, UsdShade, Sdf, Gf

stage = omni.usd.get_context().get_stage()

# ========================================
# ★ CONFIG
# ========================================
NUM_FRAMES = 1000
BATCH_ID   = int(time.time()) % 100000
OUTPUT_DIR = f"/home/kim/replicator_output/oblique_v5_batch_{BATCH_ID}"
RESOLUTION = (1280, 720)

CAMERA_PATH  = "/World/Camera"
PLATE_SHADER = "/World/Looks/Aluminum_Scratched/Shader"
BEAD_SHADER  = "/World/Looks/WeldBead_Metal/Shader"
FLOOR_SHADER = "/World/Looks/IndustrialFloor/Shader"

SURFACE_Z = 0.1
N_SPATTER = 25
N_BOLT    = 5

# 카메라
CAMERA_POS = Gf.Vec3d(-3.2, 0.4, 2.9)
QUAT_BASE  = np.array([-0.00852, 0.3683, -0.06509, -0.92739])
QUAT_RANGE = 0.05

# 시편 실제 영역
PLATE_X_MIN = -0.98277
PLATE_X_MAX =  0.98146
PLATE_Y_MIN = -0.75635
PLATE_Y_MAX =  0.73429

# 비드 exclusion zone
BEAD_CENTER_X   = -0.68942
BEAD_CENTER_Y   =  0.29385
BEAD_HALF_WIDTH =  0.018
BEAD_HALF_LEN   =  0.066

X_LEFT_MAX  = BEAD_CENTER_X - BEAD_HALF_WIDTH
X_RIGHT_MIN = BEAD_CENTER_X + BEAD_HALF_WIDTH
# ========================================


# ========================================
# 헬퍼
# ========================================
def safe_remove(path):
    p = stage.GetPrimAtPath(path)
    if p.IsValid():
        stage.RemovePrim(path)

def random_quat(base, range_val):
    noise = np.random.uniform(-range_val, range_val, 4)
    q = base + noise
    q = q / np.linalg.norm(q)
    return Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

# ========================================
# 1. Semantic Label
# ========================================
for path, label in [
    ("/World/weld_bead/node_/mesh_", "weld_bead"),
    ("/World/base_plate/node_/mesh_", "base_plate"),
]:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        add_update_semantics(prim, label)
        print(f"[OK] {path} -> {label}")
    else:
        print(f"[ERROR] NOT FOUND: {path}")

# ========================================
# 1.5. 방해 요소 생성 + Semantic Label
# ========================================
SPATTER_PATHS = []
for i in range(N_SPATTER):
    path = f"/World/Distractor_Spatter_{i}"
    safe_remove(path)
    sp = UsdGeom.Sphere.Define(stage, path)
    sp.GetRadiusAttr().Set(random.uniform(0.001, 0.004))
    add_update_semantics(sp.GetPrim(), "background")
    SPATTER_PATHS.append(path)

BOLT_PATHS = []
for i in range(N_BOLT):
    path = f"/World/Distractor_Bolt_{i}"
    safe_remove(path)
    bt = UsdGeom.Cylinder.Define(stage, path)
    bt.GetRadiusAttr().Set(random.uniform(0.005, 0.015))
    bt.GetHeightAttr().Set(random.uniform(0.010, 0.040))
    add_update_semantics(bt.GetPrim(), "background")
    BOLT_PATHS.append(path)

print(f"[OK] Distractors: spatter×{len(SPATTER_PATHS)}, bolt×{len(BOLT_PATHS)}")

# ========================================
# 2. 렌더 안정화
# ========================================
settings = carb.settings.get_settings()
settings.set("/omni/replicator/RTSubframes", 4)
print("[OK] RTSubframes = 4")

# ========================================
# 3. 조명 초기 배치
# ========================================
FIXED_LIGHTS = [
    {"pos": (4.556, -3.600, 10.913), "rot": (-10.269,  -6.755,  50.620)},
    {"pos": (-4.338, 6.238, 13.938), "rot": (-157.268,  12.207, 216.757)},
    {"pos": (-5.589, -2.871, 12.545),"rot": (  -3.007,  11.537, 124.547)},
    {"pos": (5.005,   5.885,  9.376),"rot": (   8.489,  -9.299, 306.607)},
]
for i, lt in enumerate(FIXED_LIGHTS):
    light_path = f"/World/DR_Light_{i}"
    safe_remove(light_path)
    light = UsdLux.CylinderLight.Define(stage, light_path)
    xform = UsdGeom.Xformable(light.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*lt["pos"]))
    xform.AddRotateXYZOp().Set(Gf.Vec3d(*lt["rot"]))
    light.GetLengthAttr().Set(10.0)
    light.GetRadiusAttr().Set(1.5)
    light.GetIntensityAttr().Set(77000)
    light.GetEnableColorTemperatureAttr().Set(True)
    light.GetColorTemperatureAttr().Set(4600)
print("[OK] 4 lights placed")

# ========================================
# 4. 재질 초기값 설정
# ========================================
p = stage.GetPrimAtPath(PLATE_SHADER)
if p.IsValid():
    s = UsdShade.Shader(p)
    s.CreateInput("roughness_metal_surface",  Sdf.ValueTypeNames.Float).Set(0.0)
    s.CreateInput("roughness_scratches",      Sdf.ValueTypeNames.Float).Set(0.24)
    s.CreateInput("bump_factor",              Sdf.ValueTypeNames.Float).Set(0.30)
    s.CreateInput("dirt_amount",              Sdf.ValueTypeNames.Float).Set(0.14)
    s.CreateInput("scratches_bump_factor",    Sdf.ValueTypeNames.Float).Set(0.12)
    s.CreateInput("brightness",               Sdf.ValueTypeNames.Float).Set(0.25)
    print("[OK] Plate material init")

b = stage.GetPrimAtPath(BEAD_SHADER)
if b.IsValid():
    s = UsdShade.Shader(b)
    s.CreateInput("metallic_constant",             Sdf.ValueTypeNames.Float  ).Set(0.95)
    s.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float  ).Set(0.39)
    s.CreateInput("diffuse_color_constant",        Sdf.ValueTypeNames.Color3f).Set((0.28, 0.275, 0.275))
    print("[OK] Bead material init")

f = stage.GetPrimAtPath(FLOOR_SHADER)
if f.IsValid():
    s = UsdShade.Shader(f)
    s.CreateInput("diffuse_color_constant",        Sdf.ValueTypeNames.Color3f).Set((0.15, 0.15, 0.17))
    s.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float  ).Set(0.7)
    print("[OK] Floor material init")

# ========================================
# 5. xform op 핸들 수집
# ========================================
# 카메라
cam_prim         = stage.GetPrimAtPath(CAMERA_PATH)
cam_ops          = UsdGeom.Xformable(cam_prim).GetOrderedXformOps()
cam_translate_op = next(op for op in cam_ops if op.GetOpName() == "xformOp:translate")
cam_orient_op    = next(op for op in cam_ops if op.GetOpName() == "xformOp:orient")

# 스패터
spatter_prims = [stage.GetPrimAtPath(p) for p in SPATTER_PATHS]
spatter_t_ops = []
for sp in spatter_prims:
    ops = UsdGeom.Xformable(sp).GetOrderedXformOps()
    t_op = next((op for op in ops if op.GetOpName() == "xformOp:translate"), None)
    if t_op is None:
        t_op = UsdGeom.Xformable(sp).AddTranslateOp()
    spatter_t_ops.append(t_op)

# 볼트
bolt_prims = [stage.GetPrimAtPath(p) for p in BOLT_PATHS]
bolt_t_ops = []
for bt in bolt_prims:
    ops = UsdGeom.Xformable(bt).GetOrderedXformOps()
    t_op = next((op for op in ops if op.GetOpName() == "xformOp:translate"), None)
    if t_op is None:
        t_op = UsdGeom.Xformable(bt).AddTranslateOp()
    bolt_t_ops.append(t_op)

# ========================================
# 6. Writer
# ========================================
rp = rep.create.render_product(CAMERA_PATH, RESOLUTION)
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=OUTPUT_DIR,
    rgb=True,
    semantic_segmentation=True,
    colorize_semantic_segmentation=True,
    image_output_format="png"
)
writer.attach([rp])

# ========================================
# 7. ★ Async 실행 루프 — 모든 DR 매 프레임 처리
# ========================================
async def run_replicator():
    await rep.orchestrator.step_async(delta_time=0.0)  # 초기화

    for frame_idx in range(NUM_FRAMES):

        # ── 카메라 ──
        cam_translate_op.Set(CAMERA_POS)
        cam_orient_op.Set(random_quat(QUAT_BASE, QUAT_RANGE))

        # ── 조명 (inputs: 방식) ──
        for i in range(4):
            lp = stage.GetPrimAtPath(f"/World/DR_Light_{i}")
            lp.GetAttribute("inputs:intensity").Set(random.uniform(0, 4000))
            lp.GetAttribute("inputs:colorTemperature").Set(random.uniform(6000, 8000))

        # ── 비드 재질 ──
        b = stage.GetPrimAtPath(BEAD_SHADER)
        if b.IsValid():
            s = UsdShade.Shader(b)
            s.GetInput("metallic_constant").Set(random.uniform(0.9, 1.0))
            s.GetInput("reflection_roughness_constant").Set(random.uniform(0.3, 0.5))
            s.GetInput("diffuse_color_constant").Set((
                random.uniform(0.23, 0.33),
                random.uniform(0.22, 0.32),
                random.uniform(0.22, 0.32)
            ))

        # ── 플레이트 재질 ──
        p = stage.GetPrimAtPath(PLATE_SHADER)
        if p.IsValid():
            s = UsdShade.Shader(p)
            s.GetInput("bump_factor").Set(random.uniform(0.23, 0.38))
            s.GetInput("dirt_amount").Set(random.uniform(0.00, 0.28))
            s.GetInput("scratches_bump_factor").Set(random.uniform(0.00, 0.24))
            s.GetInput("brightness").Set(random.uniform(0.10, 0.40))

        # ── 바닥 재질 ──
        f = stage.GetPrimAtPath(FLOOR_SHADER)
        if f.IsValid():
            s    = UsdShade.Shader(f)
            gray = random.uniform(0.08, 0.25)
            s.GetInput("diffuse_color_constant").Set((gray, gray, gray + 0.02))
            s.GetInput("reflection_roughness_constant").Set(random.uniform(0.5, 0.9))

        # ── 스패터 ──
        for idx, (sp_prim, t_op) in enumerate(zip(spatter_prims, spatter_t_ops)):
            if random.random() > 0.5:
                UsdGeom.Imageable(sp_prim).MakeVisible()
                x_lo, x_hi = (PLATE_X_MIN, X_LEFT_MAX) if idx % 2 == 0 else (X_RIGHT_MIN, PLATE_X_MAX)
                t_op.Set(Gf.Vec3d(
                    random.uniform(x_lo, x_hi),
                    random.uniform(PLATE_Y_MIN, PLATE_Y_MAX),
                    SURFACE_Z + random.uniform(0.001, 0.004)
                ))
            else:
                UsdGeom.Imageable(sp_prim).MakeInvisible()

        # ── 볼트 ──
        for idx, (bt_prim, t_op) in enumerate(zip(bolt_prims, bolt_t_ops)):
            if random.random() > 0.33:
                UsdGeom.Imageable(bt_prim).MakeVisible()
                x_lo, x_hi = (PLATE_X_MIN, X_LEFT_MAX) if idx % 2 == 0 else (X_RIGHT_MIN, PLATE_X_MAX)
                t_op.Set(Gf.Vec3d(
                    random.uniform(x_lo, x_hi),
                    random.uniform(PLATE_Y_MIN, PLATE_Y_MAX),
                    SURFACE_Z
                ))
            else:
                UsdGeom.Imageable(bt_prim).MakeInvisible()

        await rep.orchestrator.step_async(delta_time=0.0)
        print(f"[{frame_idx+1}/{NUM_FRAMES}] frame done")

    print(f"\n{'='*50}")
    print(f"[DONE] {NUM_FRAMES} frames -> {OUTPUT_DIR}")
    print(f"  카메라: {tuple(CAMERA_POS)} 고정")
    print(f"  orient 랜덤 범위: ±{QUAT_RANGE}")
    print(f"  방해 요소: spatter×{N_SPATTER}, bolt×{N_BOLT}")
    print(f"  매 프레임: 조명/재질/방해요소 전부 랜덤화")
    print(f"  시편 위치: 고정")
    print(f"{'='*50}")

asyncio.ensure_future(run_replicator())
