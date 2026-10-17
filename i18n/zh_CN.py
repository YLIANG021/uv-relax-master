TRANSLATIONS = {
    ('*', 'UV Relax'): 'UV 松弛',
    ('*', 'Controller'): '控制器',
    ('*', 'Transform'): '变换',
    ('*', 'Relax Selection'): '松弛所选 UV',
    ('*', 'Density Overlay'): '密度叠加',
    ('*', 'Select Interior'): '选择内部',
    ('*', 'Select Boundary'): '选择外边界',
    ('*', 'Align UV Points'): '对齐 UV 点',
    ('*', 'Select UV Region'): '选择 UV 区域',
    ('*', 'Direction'): '方向',
    ('*', 'Left'): '左对齐',
    ('*', 'Right'): '右对齐',
    ('*', 'Down'): '下对齐',
    ('*', 'Up'): '上对齐',
    ('Operator', 'left'): '左',
    ('Operator', 'right'): '右',
    ('Operator', 'down'): '下',
    ('Operator', 'up'): '上',
    ('*', 'Steps'): '迭代步数',
    ('*', 'Density Balance'): '密度平衡',
    ('*', 'Relax Axis'): '松弛轴向',
    ('*', 'XY'): '水平和垂直',
    ('*', 'X'): '仅水平',
    ('*', 'Y'): '仅垂直',
    ('*', 'Interior'): '内部',
    ('*', 'Boundary'): '边界',
    ('*', "How many refinements one island may make inside a single solve.  The island's "
           'shape is worked out rather than walked towards, so this only decides how exact '
           'that answer is; raise it on a big island whose report says it ran out'):
        '单次求解中每个 UV 岛可执行的细化次数。数值越高结果越精确；大型岛提示达到步数上限时可调高。',
    ('*', 'How much of the island\'s shape relaxation is allowed to spend in order to even '
           'out texel density, weighed against the island\'s own area so one setting behaves '
           'the same on a tightly packed island and a loosely packed one.  0 keeps pure shape '
           'fidelity and leaves the packing as the unwrap left it.  Around this default the '
           'packing evens out and the faces keep their own shape; past about 0.5 the evenness '
           'is bought from the faces themselves, which start to look skewed, and under about '
           '0.1 the packing does not even out -- some faces keep sitting far off the island\'s '
           'average density'):
        '控制岛内纹素密度均匀程度与形状保持之间的平衡。0 主要保持形状；默认值会改善密度均匀性；过高可能使面变形。',
    ('*', 'Relax both UV axes'): '同时松弛 UV 两个轴向',
    ('*', 'Relax only the horizontal UV coordinate'): '仅松弛 UV 水平坐标',
    ('*', 'Relax only the vertical UV coordinate'): '仅松弛 UV 垂直坐标',
    ('*', 'Colour each UV face by how much texel space it got compared to the average of the island it is on'):
        '按 UV 面相对所在岛平均纹素密度的偏差着色',
    ('*', 'Align selected UV points to the leftmost selected point'): '将所选 UV 点对齐到最左侧点',
    ('*', 'Align selected UV points to the rightmost selected point'): '将所选 UV 点对齐到最右侧点',
    ('*', 'Align selected UV points to the lowest selected point'): '将所选 UV 点对齐到最低点',
    ('*', 'Align selected UV points to the highest selected point'): '将所选 UV 点对齐到最高点',
    ('*', 'Select all non-boundary points'): '选择岛内所有非边界点',
    ('*', 'Select all boundary points'): '选择岛的边界点',
    ('*', 'Select the interior or boundary UV points of islands touched by the current selection.'):
        '选择当前选区所在 UV 岛的内部点或边界点',
    ('*', 'Align selected UV points to an edge of their combined bounding box.'):
        '将所选 UV 点对齐到整体边界框的一侧',
    ('Operator', 'This object has no UV map'): '此对象没有 UV 图层',
    ('Operator', 'Nothing selected in the UV editor'): 'UV 编辑器中没有选中内容',
    ('Operator', 'Nothing relaxed: the current state is already settled'):
        '未执行松弛：当前状态已达到稳定结果',
    ('Operator', 'Select at least one UV point to align'): '请至少选择一个 UV 点再对齐',
    ('Operator', 'Select at least one UV point to mark an island'):
        '请至少选择一个 UV 点以标记 UV 岛',
    ('Operator', '%d UV point(s) aligned %s'): '已将 %d 个 UV 点向%s对齐',
    ('Operator', '%d interior UV point(s) selected on %d island(s)'):
        '已选择 %d 个内部 UV 点，分布在 %d 个 UV 岛中',
    ('Operator', '%d boundary UV point(s) selected on %d island(s)'):
        '已选择 %d 个边界 UV 点，分布在 %d 个 UV 岛中',
    ('Operator', '%d UV points on %d island(s) relaxed in %.0f ms | stretch median %.3f '
                 '-> %.3f, worst 1%% %.2f -> %.2f | density CV %.3f -> %.3f, '
                 'p90/median %.2f -> %.2f, p99/median %.2f -> %.2f | '
                 'relaxed UV area %+.1f%% | %d face(s) inside out, was %d | '
                 '%d solve(s), %d refinement(s)%s'):
        '已松弛 %d 个 UV 点，涉及 %d 个 UV 岛（%.0f 毫秒）| 拉伸中位数 %.3f -> %.3f，'
        '最差 1%% %.2f -> %.2f | 密度变异系数 %.3f -> %.3f，P90/中位数 %.2f -> %.2f，'
        'P99/中位数 %.2f -> %.2f | UV 面积变化 %+.1f%% | 翻转面 %d 个（原为 %d 个）| '
        '求解 %d 次，细化 %d 次%s',
}
