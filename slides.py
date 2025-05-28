import numpy as np
import sympy as sy
import random
from manim import *
from manim_slides import Slide
import numpy as np
from scipy.stats import norm

"""
Some useful function required for the 'simple example'.
"""


def row(*args):
    """Create a symbol row (or col) vector from input arguments."""
    return sy.Matrix(args)


def generate_c(as_gamma=False):
    """Return C(x) and it's derivative."""
    # Unknowns
    s1, s2 = sy.symbols("s_1 s_2", real=True)  # s1, s2 are real values

    # Geometry

    # - Nodes
    BS = row(2, -1)
    UE = row(2, 4)

    # - Interaction points
    X1 = row(s1, s1)
    X2 = row(5, s2)

    # - Surface normals
    n1 = row(1, -1).normalized()
    n2 = row(-1, 0).normalized()

    # - Aliases
    V0 = X1 - BS
    V1 = X2 - X1
    V2 = UE - X2

    if as_gamma:
        g1 = sy.Function(r"\gamma_1", real=True)(s1, s2)
        g2 = sy.Function(r"\gamma_2", real=True)(s1, s2)
    else:
        g1 = V0.norm() / V1.norm()
        g2 = V1.norm() / V2.norm()

    # Write different equations
    eqs = [
        g1 * V1 - (V0 - 2 * V0.dot(n1) * n1),
        g2 * V2 - (V1 - 2 * V1.dot(n2) * n2),
    ]

    F = sy.Matrix.vstack(*eqs)
    f = F.norm() ** 2

    _df = sy.lambdify((s1, s2), row(f.diff(s1), f.diff(s2)))

    def df(x):
        return _df(*x).reshape(-1)

    return sy.lambdify((s1, s2), f), df




"""
Here, because I switched the background from black to white,
so I have to make default color for most things to be black (instead of white).
"""


def black(func):
    """Sets default color to black"""

    def wrapper(*args, color=BLACK, **kwargs):
        return func(*args, color=color, **kwargs)

    return wrapper


Tex = black(Tex)
Text = black(Text)
MathTex = black(MathTex)
Line = black(Line)
Dot = black(Dot)
Brace = black(Brace)
Arrow = black(Arrow)
Angle = black(Angle)


"""
Slides generation
"""


class Main(Slide,MovingCameraScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slide_no = None
        self.slide_text = None

    def write_slide_number(self, inital=1, text=Tex, animation=Write, position=ORIGIN):
        self.slide_no = inital
        self.slide_text = text(str(inital)).shift(position)
        return animation(self.slide_text)

    def update_slide_number(self, text=Tex, animation=Transform):
        self.slide_no += 1
        new_text = text(str(self.slide_no)).move_to(self.slide_text)
        return animation(self.slide_text, new_text)
    
    def next_slide_number_animation(self):
        return self.slide_number.animate(run_time=0.5).set_value(
            self.slide_number.get_value() + 1
        )

    def next_slide_title_animation(self, title):
        return Transform(
            self.slide_title,
            Text(title, color=BLACK, font_size=self.TITLE_FONT_SIZE)
            .move_to(self.slide_title)
            .align_to(self.slide_title, LEFT),
        )
    def new_clean_slide(self, title, contents=None):
        if self.mobjects_without_canvas:
            self.play(
                self.next_slide_number_animation(),
                self.next_slide_title_animation(title),
                self.wipe(
                    self.mobjects_without_canvas,
                    contents if contents else [],
                    return_animation=True,
                ),
            )
        else:
            self.play(
                self.next_slide_number_animation(),
                self.next_slide_title_animation(title),
            )

    def construct(self):
        self.camera.background_color = WHITE
        WALL_COLOR = ORANGE
        BS_COLOR = BLUE
        UE_COLOR = "#201E1E"
        GOOD_COLOR = "#28C137"
        BAD_COLOR = "#FF0000"
        IMAGE_COLOR = "#636463"
        X_COLOR = DARK_BROWN

        NW = Dot().to_corner(UL)
        NE = Dot().to_corner(UR)
        SW = Dot().to_corner(DL)
        SE = Dot().to_corner(DR)
        NL = Line(NW.get_center(), NE.get_center()).set_color(WALL_COLOR)
        SL = Line(SW.get_center(), SE.get_center()).set_color(WALL_COLOR)
        WL = Line(NW.get_center(), SW.get_center()).set_color(WALL_COLOR)
        EL = Line(NE.get_center(), SE.get_center()).set_color(WALL_COLOR)

        self.TITLE_FONT_SIZE = 36
        self.CONTENT_FONT_SIZE = 0.6 * self.TITLE_FONT_SIZE
        self.SOURCE_FONT_SIZE = 0.2 * self.TITLE_FONT_SIZE

        # Mutable variables

        self.slide_number = Integer(1).set_color(BLACK).to_corner(DR)
        self.slide_title = Text(
            "Contents", color=BLACK, font_size=self.TITLE_FONT_SIZE
        ).to_corner(UL)
        self.add_to_canvas(slide_number=self.slide_number, slide_title=self.slide_title)

        slide_no_pos = SE.shift(0.15 * RIGHT + 0.2 * DOWN).get_center()

        # TeX Preamble
        tex_template = TexTemplate()
        tex_template.add_to_preamble(
            r"""
\usepackage{fontawesome5}
\usepackage{siunitx}
\DeclareSIQualifier\wattref{W}
\DeclareSIUnit\dbw{\decibel\wattref}
\usepackage{amsmath,amssymb,amsfonts,mathtools}
\newcommand{\bs}{\boldsymbol}
\newcommand{\scp}[3][]{#1\langle #2, #3 #1\rangle}
\newcommand{\bb}{\mathbb}
\newcommand{\cl}{\mathcal}
"""
        )

        # Slide: Title
        logo= ImageMobject("Images/logo.png")
        logo.scale(0.05)
        logo.to_corner(DL, buff=0.2)
        self.add(logo)
        line1 = Text("Leveraging Posterior Uncertainty of Treatment Effects", color=BLACK).scale(0.5)
        line2 = Text("in Bayesian Response-Adaptive Group Sequential Designs", color=BLACK).scale(0.5)
        title = VGroup(line1, line2).arrange(DOWN, buff=0.1).move_to(ORIGIN)
        author = (
            Text("Corey Voller", color=BLACK)
            .scale(0.4)
            .next_to(title, DOWN)
        )
        date = (
            Text("June 5th 2025", color=BLACK)
            .scale(0.4)
            .next_to(author, DOWN)
        )

        self.play(FadeIn(title),FadeIn(author,direction=DOWN),FadeIn(date,direction=DOWN),self.write_slide_number(position=slide_no_pos))
        self.next_slide()

        # Slide: room
        slide_number = self.update_slide_number()
        self.play(FadeOut(title),FadeOut(date),FadeOut(author),slide_number)
        

        title = Text("Introduction & Background", font_size=48, color=BLACK)
        title.to_corner(UL)  # Positions it at the top center of the screen
        self.play(Write(title))

        # Import svg
        cohort = VGroup()
        for i in range(-2, 3):
            for j in range(-1, 3):
                cohort.add(
                    SVGMobject("Images/person.svg", fill_color=BLACK,opacity=1)
                    .scale(0.25)
                    .shift(i * UP + j * LEFT)
                )
        cohort.shift(3 * LEFT)  # Move entire cohort left

        # Add and fade in cohort
        self.add(cohort)
        self.play(FadeIn(cohort))

        self.next_slide()



        label_x = 1.5

        treatment_label = Text("Treatment").shift(UP * 2)
        control_label = Text("Control").shift(DOWN * 2)

        treatment_label.align_to(np.array([label_x, 0, 0]), LEFT)
        control_label.align_to(np.array([label_x, 0, 0]), LEFT)

        self.add(treatment_label, control_label)
        self.play(FadeIn(treatment_label), FadeIn(control_label))

        # Starting point: middle right edge of the cohort
        start_point = cohort.get_right()

        # Arrows going from the middle right of cohort outward forming a sideways V
        arrow_to_treatment = Arrow(
            start=start_point,
            end=treatment_label.get_left() + LEFT * 0.3,
            buff=0,
            stroke_width=3
        )

        arrow_to_control = Arrow(
            start=start_point,
            end=control_label.get_left() + LEFT * 0.3,
            buff=0,
            stroke_width=3
        )

        # Labels
        p_label = MathTex("p", font_size=28, color=BLACK)
        one_minus_p_label = MathTex("1 - p", font_size=28, color=BLACK)

        # Position p_label above the first arrow
        p_label.next_to(arrow_to_treatment, UP, buff=0.2)

        # Position 1 - p label below the second arrow
        one_minus_p_label.next_to(arrow_to_control, DOWN, buff=0.2)

        self.play(GrowArrow(arrow_to_treatment),
                  GrowArrow(arrow_to_control),
                  Write(p_label),
                  Write(one_minus_p_label))
        self.wait(0.5)

        self.next_slide()

        for idx, person in enumerate(cohort):
            # Clear existing fills and strokes on all submobjects inside each SVG
            for submob in person.submobjects:
                submob.set_fill(opacity=0)
                submob.set_stroke(width=0)
    
            # Then apply desired fill color on the whole SVG object

            if idx < 10:
                person.set_fill(RED, opacity=1)
            else:
                person.set_fill(BLUE, opacity=1)

        self.play(
            *[person.animate.set_fill(person.get_fill_color(), opacity=1) for person in cohort]
        )

        self.next_slide()
        # Step 1: Camera pans to the right
        self.play(self.camera.frame.animate.shift(RIGHT * 6))  # Adjust shift as needed

# Step 2: Add the new title
        interim_text = Text("Interim Analysis I", font_size=28, color=BLACK)
        interim_text.move_to(self.camera.frame.get_right() + LEFT * 4)
        self.play(FadeIn(interim_text))

# Step 3: Draw arrows from Treatment and Control to Interim Analysis I
        arrow_from_treatment = Arrow(
            start=treatment_label.get_right(),
            end=interim_text.get_left() + UP * 0.5,
            buff=0.1,
            stroke_width=3
        )

        arrow_from_control = Arrow(
            start=control_label.get_right(),
            end=interim_text.get_left() + DOWN * 0.5,
            buff=0.1,
            stroke_width=3
        )

        self.play(GrowArrow(arrow_from_treatment), GrowArrow(arrow_from_control))
        self.next_slide()  
        p = 0.5  # Example probability
        chart = BarChart(
            values=[p * 100, (1 - p) * 100],  # Convert to percentages
            y_range=[0, 100, 10],
            y_length=3,
            x_length=2.5,
            bar_names=["T", "C"],
            y_axis_config={
                "decimal_number_config": {
                    "unit": "\\%",
                    "num_decimal_places": 0,
                    "color": BLACK
               },
               "color": BLACK
           },
           x_axis_config={"color": BLACK}
        )
        #c_bar_lbls = chart.get_bar_labels(font_size=24,color=BLACK)
        bar_names_labels = VGroup()
        #for name, bar in zip(chart.bar_names,chart.bars):
        #        label = Text(name, font_size=24,color=BLACK)
        #        label.next_to(bar,DOWN,buff=0.2)
        #        c_bar_lbls.add(label)

        #self.add(c_bar_lbls)
        chart.move_to(interim_text.get_bottom()+ DOWN*1.5)
        y_label = Text("Allocation Prob (%)", font_size=24).rotate(PI / 2)
        y_label.next_to(chart.y_axis, LEFT, buff=0.1)  # Position to the left of Y-axis
        self.add(y_label)

        h_lines = VGroup(*[
           Line(chart.c2p(0, x), chart.c2p(3, x), stroke_width=1)
           for x in range(0, 110, 10)
        ])
        h_lines.set_opacity(0)
        h_r, s_r = chart.bars
        s_r.set_color(RED)

        chart_group = VGroup(chart, h_lines, y_label)
        chart_group.scale(0.7)
        self.add(chart_group)
        #self.add(chart, h_lines, *chart.bars)
        self.chart = chart
        self.h_lines = h_lines

        # Decimal values for percentages
        h_dn = DecimalNumber(p * 100, color=BLACK,font_size=24)
        s_dn = DecimalNumber((1 - p) * 100, color=BLACK,font_size=24)
        hsl_variables = VGroup(h_dn, s_dn)

        def chart_updater(mob: BarChart):
            hb, sb = mob.bars
            new_vals = [h_dn.get_value(), s_dn.get_value()]
            mob.change_bar_values(new_vals)

    # Position & update color of number labels
            h_dn.next_to(hb, UP, buff=0.1)
            s_dn.next_to(sb, UP, buff=0.1)
            h_dn.set_color(BLACK)
            s_dn.set_color(BLACK)

        chart.add_updater(chart_updater, call_updater=True)
        self.add(chart, hsl_variables)

# Animate change in probability
        self.play(
            ChangeDecimalToValue(h_dn, 70),
            ChangeDecimalToValue(s_dn, 30),
         run_time=2
        )
        self.play(
            ChangeDecimalToValue(h_dn, 40),
            ChangeDecimalToValue(s_dn, 60),
            run_time=2
        )
        self.wait()
        self.next_slide()
        #self.mobjects.remove(self.camera.frame)        
        #keep = [title, logo, slide_number]
        #
        #for mob in self.mobjects[:]:  # [:] to clone the list since we're modifying it
        #    if mob not in keep:
        #        self.remove(mob)
        #self.next_slide()

        # Boundary
        # Define k values and boundaries (boundaries start at k=1)
        k_values = np.array([0, 1, 2, 3, 4, 5])
        a_crit = np.array([-1.61511306, -0.07126633, 0.81610852, 1.46393433, 1.986610])  # No k=0 boundary
        b_crit = np.array([4.442196, 3.141107, 2.564703, 2.221098, 1.986610])  # No k=0 boundary
        a_crit_ext = np.insert(a_crit, 0, a_crit[0])
        b_crit_ext = np.insert(b_crit, 0, b_crit[0])
        # Observed paths (starting from k=0, y=0)
        observed_red_cross = np.array([0, 1.5, 2.3, 2.7, 4.5])  # Crosses upper boundary
        observed_green_cross = np.array([0, -1.2, -1.8, -2.2, -2.5])  # Crosses lower boundary
        observed_no_cross = np.array([0,1.8, 1, 1.5, 1.8, 2.1])  # Stays within bounds
                # Create axes
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[-3, 5, 1],
            axis_config={"color": BLACK},
            x_axis_config={
                "include_numbers": True,
                "numbers_to_include": [1, 2, 3, 4, 5],
                "decimal_number_config": {"num_decimal_places":0,
                "color": BLACK}
            },
            y_axis_config={
                "include_numbers": True,
                "decimal_number_config": {"color": BLACK}
            }
        )
        axes.next_to(cohort, DOWN, buff=1.5)
        x = axes.get_x_axis()
        x.numbers.set_color(BLACK)
        x_label = Tex("Analysis (k)").next_to(axes.x_axis, DOWN, buff=0.5)
        y_label = MathTex("Z_k").next_to(axes.y_axis, LEFT)
        
        self.play(
            self.camera.frame.animate.move_to(axes)
        )
        self.wait()

        self.play(Create(axes),Write(x_label), Write(y_label))

        # Plot boundaries (starting at k=1)
        a_crit_line = axes.plot_line_graph(
            x_values=k_values[1:], y_values=a_crit, add_vertex_dots=False, line_color=GREEN, stroke_width=4
        )
        b_crit_line = axes.plot_line_graph(
            x_values=k_values[1:], y_values=b_crit, add_vertex_dots=False, line_color=RED, stroke_width=4
        )
        self.play(Create(a_crit_line), Create(b_crit_line))
        # Labels on graph
        reject_text = MathTex(r"\text{Reject } H_0", font_size=28)
        reject_text.move_to(axes.c2p(3, b_crit[2]) + UP * 0.5)
        self.play(Write(reject_text))

        accept_text = MathTex(r"\text{Accept } H_0", font_size=28)
        accept_text.move_to(axes.c2p(4, a_crit[2]))
        self.play(Write(accept_text))
        continue_text = MathTex(r"\text{Continue}", font_size=28)
        continue_text.move_to(axes.c2p(1, 2) + UP * 0.5)
        self.play(Write(continue_text))

        upper_curve = axes.plot_line_graph(
            x_values=k_values[1:], y_values=b_crit,
            add_vertex_dots=False, line_color=RED, stroke_width=0  # Invisible curve
        )

        lower_curve = axes.plot_line_graph(
            x_values=k_values[1:], y_values=a_crit,
            add_vertex_dots=False, line_color=GREEN, stroke_width=0  # Invisible curve
        )
        fill_tracker = ValueTracker(0.0)

        def get_blue_region():
            lower_points = [axes.c2p(x, a) for x, a in zip(k_values[1:], a_crit)]
            upper_points = [axes.c2p(x, a + (b - a) * fill_tracker.get_value())
                    for x, a, b in zip(k_values[1:], a_crit, b_crit)]
            return Polygon(*lower_points, *reversed(upper_points),
                   color=BLUE, fill_opacity=0.2, stroke_opacity=0)

        blue_region = always_redraw(get_blue_region)
        self.add(blue_region)

        def get_left_blue_strip():
            top_y = axes.y_range[1]
            bottom_y = axes.y_range[0]
            k0 = 0
            k1 = 1
            return Polygon(
                axes.c2p(k0, bottom_y),
                axes.c2p(k1, bottom_y),
                axes.c2p(k1, bottom_y + (top_y - bottom_y) * fill_tracker.get_value()),
                axes.c2p(k0, bottom_y + (top_y - bottom_y) * fill_tracker.get_value()),
                color=BLUE, fill_opacity=0.2, stroke_opacity=0
            )

        left_blue_strip = always_redraw(get_left_blue_strip)
        self.add(left_blue_strip)

        def get_red_region():
            top_y = axes.y_range[1]
            upper_points = [axes.c2p(x, b) for x, b in zip(k_values[1:], b_crit)]
            top_points = [axes.c2p(x, b + (top_y - b) * fill_tracker.get_value())
                  for x, b in zip(k_values[1:], b_crit)]
            return Polygon(*upper_points, *reversed(top_points),
                   color=RED, fill_opacity=0.3, stroke_opacity=0)

        red_region = always_redraw(get_red_region)
        self.add(red_region)

        def get_green_region():
            bottom_y = axes.y_range[0]
            lower_points = [axes.c2p(x, a) for x, a in zip(k_values[1:], a_crit)]
            bottom_points = [axes.c2p(x, a + (bottom_y - a) * fill_tracker.get_value())
                     for x, a in zip(k_values[1:], a_crit)]
            return Polygon(*lower_points, *reversed(bottom_points),
                   color=PURE_GREEN, fill_opacity=0.3, stroke_opacity=0)

        green_region = always_redraw(get_green_region)
        self.add(green_region)
        # Add shaded regions in the correct back-to-front order
        #self.add(red_region, green_region, blue_region)
        self.play(fill_tracker.animate.set_value(1.0), run_time=2)
        self.wait()
        self.next_slide()

        boundary_group = VGroup(axes, a_crit_line, b_crit_line, reject_text, accept_text, continue_text,
                        blue_region, red_region, green_region, left_blue_strip, x_label, y_label)
        everything = VGroup(cohort, treatment_label, control_label, interim_text, chart_group, hsl_variables, boundary_group)
        target_center = everything.get_center()
        target_width = everything.width
        target_height = everything.height
        zoom_out_factor = max(target_width / config.frame_width, target_height / config.frame_height)
        self.play(
            self.camera.frame.animate.set(width=zoom_out_factor * config.frame_width).move_to(target_center),
            run_time=1
        )
        #self.mobjects.remove(self.camera.frame)
        keep = [title, logo, slide_number,axes]

        for mob in self.mobjects[:]:  # [:] to clone the list since we're modifying it
            if mob not in keep:
                self.remove(mob)
        
        title2 = Text("Leveraging Uncertainty", font_size=48, color=BLACK)
        title2.to_corner(UL)  # Positions it at the top center of the screen
        mu = 1
        sigma = 1

        # Create the new axes
        newaxes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            x_length=6,
            y_length=5,
            axis_config={"color": BLACK},
        )
        x_label = MathTex(r"\theta").scale(0.6).next_to(newaxes.x_axis, DOWN, buff=0.5)
        y_label = Tex(r"\text{Density}").scale(0.6).next_to(newaxes.y_axis, LEFT, buff=0.1)

# Normal PDF function
        def normal_pdf(x):
           return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        graph = newaxes.plot(normal_pdf, color=RED)

# Transform the axes
        self.play(Transform(axes, newaxes))  # 'axes' is assumed to be defined earlier
        self.add(x_label, y_label, graph)
        self.play(Write(x_label), Write(y_label))
        self.wait()

        self.play(
            self.camera.frame.animate.set(width=config.frame_width).move_to(ORIGIN),
            run_time=2  # Optional: adjust duration
        )
        slide_number = self.update_slide_number()
        self.play(Transform(title, title2),slide_number)

# Group everything to scale and shift
        graph_group = VGroup(newaxes, x_label, y_label, graph)
        self.remove(axes)
        self.play(graph_group.animate.scale(0.8).shift(RIGHT * 3), run_time=1.5)

        norm_data = MathTex(
        r"\text{Data } = \begin{cases}"
        r"X_{i,1}\sim N(\mu_1,\sigma^2) \\"
        r"X_{i,2}\sim N(\mu_2,\sigma^2)"
        r"\end{cases}",
        font_size=28,
        color=BLACK
        )

        latex_eq = MathTex(
        r"L(\theta) = \begin{cases}"
        r"I_1 + a^{\theta/\delta}I_2 & \text{if } \theta \geq 0 \\"
        r"I_2 + a^{-\theta/\delta}I_1 & \text{if } \theta \leq 0"
        r"\end{cases}",
        font_size=28,
        color=BLACK
        )

# Group equations
        text_group = VGroup(norm_data).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        text_group.to_corner(UL)
        text_group.shift(DOWN*1 + RIGHT*0.8)
        self.play(Write(text_group))
        self.wait()
        self.next_slide()

# Add in priors
        prior_group = MathTex(
        r"\text{Priors } = \begin{cases}"
        r"\theta \sim N(\mu_1 - \mu_2,\sigma_1^2 + \sigma_2^2) \\"
        r"\mu_j \sim N(\mu_j,\sigma_j^2)"
        r"\end{cases}",
        font_size=28,
        color=BLACK
        )

        prior_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        prior_group.next_to(text_group, DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(Write(prior_group))
        self.wait()
        self.next_slide()
        latex_eq.next_to(prior_group,DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(Write(latex_eq))
        self.next_slide()
        ratio_text = MathTex(r"\text{Ratio }= \frac{I_1}{I_2} = a^{\theta/2\delta}", font_size=28,color=BLACK)
        ratio_text.next_to(latex_eq,DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(Write(ratio_text))
        self.next_slide()
# Add vertical line animation
        current_x = ValueTracker(mu)

        vertical_line = always_redraw(
         lambda: newaxes.get_vertical_line(
             newaxes.coords_to_point(
                 current_x.get_value(),
                 normal_pdf(current_x.get_value())
             ),
             color=BLACK,
             stroke_width=3,
           )
        )
        self.add(vertical_line)
        self.play(current_x.animate.set_value(mu + 1), run_time=3)
        self.play(current_x.animate.set_value(mu - 1), run_time=3)
        self.wait()

        # Create a static vertical line at mu to flash
        # Create a copy of the current vertical line at mu for indication
        highlight_line = newaxes.get_vertical_line(
            newaxes.coords_to_point(mu, normal_pdf(mu)),
            color=BLACK,
            stroke_width=6
        )

        self.play(Indicate(highlight_line, scale_factor=1.2, color=YELLOW))
        self.wait()
        self.wait()

# Add and animate shaded area
        a = ValueTracker(mu - 1)
        b = ValueTracker(mu - 1)

        area = always_redraw(
           lambda: newaxes.get_area(
               graph,
               x_range=[a.get_value(), b.get_value()],
               color=BLUE,
               opacity=0.5,
           )
        )

        self.add(area)
        self.play(a.animate.set_value(mu - 2), b.animate.set_value(mu + 2), run_time=3)
        self.play(a.animate.set_value(mu - 1), b.animate.set_value(mu + 1), run_time=3)
        self.wait()