import rerun.blueprint as rrb
import rerun as rr

blue_print = rrb.Blueprint(
    rrb.Horizontal(
        rrb.BarChartView(),
        rrb.Vertical(rrb.Spatial2DView(), rrb.Spatial3DView())
    )
)


rr.init("ropePush", spawn=True, default_blueprint=blue_print)