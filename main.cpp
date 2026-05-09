#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

// ── grid matches window aspect ratio so cells are square ─────────────────────
constexpr int WIN_W = 1400;
constexpr int WIN_H = 900;

// NX:NY = WIN_W:WIN_H = 14:9  →  NX=280, NY=180  (cells are exactly 5×5 px)
constexpr int NX = 280;
constexpr int NY = 180;

constexpr float CELL = float(WIN_W) / NX;   // == WIN_H/NY == 5.0

constexpr int   ITER = 14;
constexpr float DT   = 0.08f;

float flowSpeed   = 2.5f;
float densityFade = 0.992f;
float viscosity   = 0.00001f;
float diffusion   = 0.00001f;
float brushRadius = 1.5f;

inline int IX(int x, int y) { return x + y * NX; }

std::vector<float> u(NX*NY), v(NX*NY), uPrev(NX*NY), vPrev(NX*NY);
std::vector<float> dens(NX*NY), densPrev(NX*NY);
// pressure reused from project() — no extra solve needed
std::vector<float> pressure(NX*NY), pdiv(NX*NY);
std::vector<bool>  solid(NX*NY, false);

// ── modes ─────────────────────────────────────────────────────────────────────
enum class Mode { Density, Vorticity, Pressure };
Mode vizMode    = Mode::Density;
bool showStreams = false;
bool windTunnel = false;

// adaptive range: EMA of per-frame max so colours don't blow out or go flat
float emaVort = 1.f, emaPres = 1.f;
constexpr float EMA_A = 0.05f;   // smoothing factor (higher = faster adapt)

// ── boundary ──────────────────────────────────────────────────────────────────
void apply_solid_bnd(int b, std::vector<float>& x) {
    for (int j = 1; j < NY-1; j++) {
        for (int i = 1; i < NX-1; i++) {
            if (!solid[IX(i,j)]) continue;
            x[IX(i,j)] = 0;
            if (b == 1) {
                if (!solid[IX(i-1,j)]) x[IX(i-1,j)] = 0;
                if (!solid[IX(i+1,j)]) x[IX(i+1,j)] = 0;
            } else if (b == 2) {
                if (!solid[IX(i,j-1)]) x[IX(i,j-1)] = 0;
                if (!solid[IX(i,j+1)]) x[IX(i,j+1)] = 0;
            }
        }
    }
}

void set_bnd(int b, std::vector<float>& x) {
    for (int i = 1; i < NX-1; i++) {
        x[IX(0,    i % NY)] = b==1 ? -x[IX(1,    i % NY)] : x[IX(1,    i % NY)];
        x[IX(NX-1, i % NY)] = b==1 ? -x[IX(NX-2, i % NY)] : x[IX(NX-2, i % NY)];
    }
    for (int i = 1; i < NX-1; i++) {
        x[IX(i, 0)]    = b==2 ? -x[IX(i, 1)]    : x[IX(i, 1)];
        x[IX(i, NY-1)] = b==2 ? -x[IX(i, NY-2)] : x[IX(i, NY-2)];
    }
    apply_solid_bnd(b, x);
}

void lin_solve(int b, std::vector<float>& x, std::vector<float>& x0,
               float a, float c) {
    float inv_c = 1.f / c;
    for (int k = 0; k < ITER; k++) {
        for (int j = 1; j < NY-1; j++)
            for (int i = 1; i < NX-1; i++) {
                if (solid[IX(i,j)]) { x[IX(i,j)] = 0; continue; }
                x[IX(i,j)] = (x0[IX(i,j)] + a*(
                    x[IX(i+1,j)] + x[IX(i-1,j)] +
                    x[IX(i,j+1)] + x[IX(i,j-1)]
                )) * inv_c;
            }
        set_bnd(b, x);
    }
}

void diffuse(int b, std::vector<float>& x, std::vector<float>& x0, float diff) {
    // use geometric mean of NX,NY for diffusion scale
    float scale = float((NX-2 + NY-2)) * 0.5f;
    float a = DT * diff * scale * scale;
    lin_solve(b, x, x0, a, 1+4*a);
}

// project stores pressure in global ::pressure for reuse in pressure view
void project(std::vector<float>& pu, std::vector<float>& pv) {
    float inv_avg = 2.f / float(NX + NY);
    for (int j = 1; j < NY-1; j++)
        for (int i = 1; i < NX-1; i++) {
            pdiv[IX(i,j)] = -0.5f*(
                pu[IX(i+1,j)] - pu[IX(i-1,j)] +
                pv[IX(i,j+1)] - pv[IX(i,j-1)]
            ) * inv_avg;
            pressure[IX(i,j)] = 0;
        }
    set_bnd(0, pdiv); set_bnd(0, pressure);
    lin_solve(0, pressure, pdiv, 1, 4);

    float half_avg = float(NX + NY) * 0.5f;
    for (int j = 1; j < NY-1; j++)
        for (int i = 1; i < NX-1; i++) {
            if (solid[IX(i,j)]) { pu[IX(i,j)] = pv[IX(i,j)] = 0; continue; }
            pu[IX(i,j)] -= 0.5f * half_avg * (pressure[IX(i+1,j)] - pressure[IX(i-1,j)]);
            pv[IX(i,j)] -= 0.5f * half_avg * (pressure[IX(i,j+1)] - pressure[IX(i,j-1)]);
        }
    set_bnd(1, pu); set_bnd(2, pv);
}

void advect(int b, std::vector<float>& d, std::vector<float>& d0,
            std::vector<float>& au, std::vector<float>& av) {
    float dt0u = DT * float(NX);
    float dt0v = DT * float(NY);
    for (int j = 1; j < NY-1; j++)
        for (int i = 1; i < NX-1; i++) {
            if (solid[IX(i,j)]) { d[IX(i,j)] = 0; continue; }
            float x = std::clamp(i - dt0u*au[IX(i,j)], 0.5f, float(NX)-1.5f);
            float y = std::clamp(j - dt0v*av[IX(i,j)], 0.5f, float(NY)-1.5f);
            int i0=int(x), i1=i0+1, j0=int(y), j1=j0+1;
            float s1=x-i0, s0=1-s1, t1=y-j0, t0=1-t1;
            auto samp=[&](int si,int sj){ return solid[IX(si,sj)]?0.f:d0[IX(si,sj)]; };
            d[IX(i,j)] = s0*(t0*samp(i0,j0)+t1*samp(i0,j1))
                       + s1*(t0*samp(i1,j0)+t1*samp(i1,j1));
        }
    set_bnd(b, d);
}

void velocity_step() {
    diffuse(1, uPrev, u, viscosity);
    diffuse(2, vPrev, v, viscosity);
    project(uPrev, vPrev);
    advect(1, u, uPrev, uPrev, vPrev);
    advect(2, v, vPrev, uPrev, vPrev);
    project(u, v);
}

void density_step() {
    diffuse(0, densPrev, dens, diffusion);
    advect(0, dens, densPrev, u, v);
    for (auto& d : dens) d *= densityFade;
}

void paint(int gx, int gy, bool draw) {
    int r = int(brushRadius);
    for (int dy=-r; dy<=r; dy++)
        for (int dx=-r; dx<=r; dx++) {
            if (dx*dx+dy*dy > r*r) continue;
            int px=gx+dx, py=gy+dy;
            if (px>1&&px<NX-2&&py>1&&py<NY-2)
                solid[IX(px,py)] = draw;
        }
}

// ── colourmaps ────────────────────────────────────────────────────────────────
// All take a value in [-1,1] or [0,1] already normalised by caller

// Density: dark navy → cyan → white
sf::Color densityColor(float d) {
    d = std::clamp(d, 0.f, 1.f);
    return {uint8_t(d*80), uint8_t(d*200), uint8_t(40+d*215)};
}

// Vorticity: blue (CCW) ↔ black (0) ↔ red (CW), gamma-brightened
sf::Color vorticityColor(float t) {
    // t in [-1,1]
    float mag = std::pow(std::abs(t), 0.5f);  // gamma brighten mid-range
    if (t >= 0)
        return {uint8_t(255*mag), uint8_t(30*(1-mag)), uint8_t(80*(1-mag))};
    else
        return {uint8_t(80*(1-mag)), uint8_t(30*(1-mag)), uint8_t(255*mag)};
}

// Pressure: cool (blue) → neutral (black) → warm (orange-red)
sf::Color pressureColor(float t) {
    // t in [-1,1]
    float mag = std::pow(std::abs(t), 0.6f);
    if (t >= 0)   // high pressure: warm orange
        return {uint8_t(255*mag), uint8_t(120*mag*(1-mag*0.5f)), uint8_t(0)};
    else           // low pressure: cool blue-purple
        return {uint8_t(60*mag), uint8_t(20*mag), uint8_t(255*mag)};
}

// ── slider ────────────────────────────────────────────────────────────────────
struct Slider { float *val; float min,max,y; const char *label; int decimals; };

int main() {
    sf::RenderWindow win(sf::VideoMode({WIN_W, WIN_H}), "Airflow Simulator");
    win.setFramerateLimit(60);

    sf::Font font;
    if (!font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
        return 1;

    std::vector<Slider> sliders = {
        {&flowSpeed,   0.5f,  8.f,    55,  "Flow Speed",    2},
        {&densityFade, 0.95f, 0.999f, 110, "Smoke Persist", 3},
        {&viscosity,   0.f,   0.001f, 165, "Viscosity",     5},
        {&diffusion,   0.f,   0.001f, 220, "Diffusion",     5},
        {&brushRadius, 0.5f,  8.f,    275, "Brush Size",    1},
    };

    bool showPanel = false;

    sf::VertexArray grid(sf::PrimitiveType::Triangles, NX*NY*6);

    sf::Text hint(font,
        "P=panel  S=streamlines  V=vorticity  Q=pressure  W=wind tunnel  C=clear",
        11);
    hint.setFillColor(sf::Color(180,180,180,200));
    hint.setPosition({8.f, float(WIN_H)-20.f});

    while (win.isOpen()) {

        // ── events ────────────────────────────────────────────────────────────
        while (auto e = win.pollEvent()) {
            if (e->is<sf::Event::Closed>()) win.close();
            if (auto* k = e->getIf<sf::Event::KeyPressed>()) {
                switch (k->code) {
                case sf::Keyboard::Key::P: showPanel  = !showPanel;  break;
                case sf::Keyboard::Key::S: showStreams = !showStreams; break;
                case sf::Keyboard::Key::V:
                    vizMode=(vizMode==Mode::Vorticity)?Mode::Density:Mode::Vorticity; break;
                case sf::Keyboard::Key::Q:
                    vizMode=(vizMode==Mode::Pressure)?Mode::Density:Mode::Pressure;   break;
                case sf::Keyboard::Key::W:
                    windTunnel=!windTunnel;
                    for(int i=0;i<NX;i++){
                        solid[IX(i,1)]    = windTunnel;
                        solid[IX(i,NY-2)] = windTunnel;
                    }
                    break;
                case sf::Keyboard::Key::C:
                    std::fill(solid.begin(),solid.end(),false);
                    if(windTunnel) for(int i=0;i<NX;i++){
                        solid[IX(i,1)]=solid[IX(i,NY-2)]=true;}
                    std::fill(dens.begin(), dens.end(), 0);
                    std::fill(u.begin(),    u.end(),    0);
                    std::fill(v.begin(),    v.end(),    0);
                    break;
                default: break;
                }
            }
        }

        auto m  = sf::Mouse::getPosition(win);
        int  gx = int(m.x / CELL);
        int  gy = int(m.y / CELL);

        bool overPanel = showPanel && m.x < 240 && m.y < 330;
        if (!overPanel) {
            if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left))  paint(gx,gy,true);
            if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Right)) paint(gx,gy,false);
        }

        // inject left-edge flow
        for (int j=1; j<NY-1; j++) {
            u[IX(1,j)]    = flowSpeed;
            dens[IX(1,j)] = 1.f;
        }

        velocity_step();
        density_step();
        // pressure is already updated inside project() — free!

        // ── update adaptive EMA ranges ─────────────────────────────────────
        if (vizMode == Mode::Vorticity) {
            float frameMax = 1e-6f;
            for (int j=1;j<NY-1;j++) for (int i=1;i<NX-1;i++) {
                float w = (v[IX(i+1,j)]-v[IX(i-1,j)]
                          -u[IX(i,j+1)]+u[IX(i,j-1)]) * 0.5f;
                frameMax = std::max(frameMax, std::abs(w));
            }
            emaVort = emaVort*(1-EMA_A) + frameMax*EMA_A;
        }
        if (vizMode == Mode::Pressure) {
            float frameMax = 1e-6f;
            for (auto p : pressure) frameMax = std::max(frameMax, std::abs(p));
            emaPres = emaPres*(1-EMA_A) + frameMax*EMA_A;
        }

        // ── build vertex array ─────────────────────────────────────────────
        for (int j=0; j<NY; j++) {
            for (int i=0; i<NX; i++) {
                float x0=i*CELL, y0=j*CELL, x1=x0+CELL+1, y1=y0+CELL+1;

                sf::Color col;
                if (solid[IX(i,j)]) {
                    col = sf::Color(220,60,60);
                } else {
                    switch (vizMode) {
                    case Mode::Density:
                        col = densityColor(dens[IX(i,j)]);
                        break;
                    case Mode::Vorticity: {
                        float w = 0;
                        if (i>0&&i<NX-1&&j>0&&j<NY-1)
                            w=(v[IX(i+1,j)]-v[IX(i-1,j)]
                              -u[IX(i,j+1)]+u[IX(i,j-1)])*0.5f;
                        col = vorticityColor(w / emaVort);
                        break;
                    }
                    case Mode::Pressure:
                        col = pressureColor(pressure[IX(i,j)] / emaPres);
                        break;
                    }
                }

                int base=(j*NX+i)*6;
                sf::Vertex tl{{x0,y0},col}, tr{{x1,y0},col};
                sf::Vertex bl{{x0,y1},col}, br{{x1,y1},col};
                grid[base+0]=tl; grid[base+1]=tr; grid[base+2]=br;
                grid[base+3]=tl; grid[base+4]=br; grid[base+5]=bl;
            }
        }

        win.clear(sf::Color(10,10,20));
        win.draw(grid);

        // ── streamline arrows ─────────────────────────────────────────────
        if (showStreams) {
            constexpr int   STEP = 14;
            constexpr float ALEN = 4.f;
            sf::VertexArray lines(sf::PrimitiveType::Lines);

            for (int j=STEP/2; j<NY; j+=STEP)
                for (int i=STEP/2; i<NX; i+=STEP) {
                    if (solid[IX(i,j)]) continue;
                    float uu=u[IX(i,j)], vv=v[IX(i,j)];
                    float spd=std::sqrt(uu*uu+vv*vv);
                    if (spd<0.01f) continue;
                    float nx=uu/spd, ny=vv/spd;
                    float cx=(i+0.5f)*CELL, cy=(j+0.5f)*CELL;
                    float ex=cx+nx*ALEN*2, ey=cy+ny*ALEN*2;
                    uint8_t br=uint8_t(std::min(255.f,spd/flowSpeed*220));
                    sf::Color ac(br,br,255,190);
                    lines.append({{cx,cy},ac}); lines.append({{ex,ey},ac});
                    float lx=nx*ALEN-ny*ALEN*0.5f, ly=ny*ALEN+nx*ALEN*0.5f;
                    lines.append({{ex,ey},ac}); lines.append({{ex-lx,ey-ly},ac});
                    float rx=nx*ALEN+ny*ALEN*0.5f, ry=ny*ALEN-nx*ALEN*0.5f;
                    lines.append({{ex,ey},ac}); lines.append({{ex-rx,ey-ry},ac});
                }
            win.draw(lines);
        }

        win.draw(hint);

        // ── control panel ─────────────────────────────────────────────────
        if (showPanel) {
            constexpr float PX=14, BAR_W=170, BAR_H=4;
            sf::RectangleShape panel({240,330});
            panel.setFillColor(sf::Color(15,15,25,230));
            panel.setOutlineColor(sf::Color(60,60,100));
            panel.setOutlineThickness(1);
            win.draw(panel);

            sf::Text title(font,"CONTROLS",13);
            title.setStyle(sf::Text::Bold);
            title.setFillColor(sf::Color(140,180,255));
            title.setPosition({PX,12}); win.draw(title);

            const char* modeStr =
                vizMode==Mode::Vorticity ? "view: VORTICITY" :
                vizMode==Mode::Pressure  ? "view: PRESSURE"  : "view: DENSITY";
            sf::Text modeLabel(font,modeStr,10);
            modeLabel.setFillColor(sf::Color(100,220,150));
            modeLabel.setPosition({PX,30}); win.draw(modeLabel);

            sf::Text wtLabel(font,windTunnel?"tunnel: ON":"tunnel: OFF",10);
            wtLabel.setFillColor(windTunnel?sf::Color(100,220,100):sf::Color(140,140,160));
            wtLabel.setPosition({PX+110,30}); win.draw(wtLabel);

            for (auto& s : sliders) {
                sf::Text lbl(font,s.label,11);
                lbl.setFillColor(sf::Color(190,210,255));
                lbl.setPosition({PX,s.y-18}); win.draw(lbl);

                std::ostringstream oss;
                oss<<std::fixed<<std::setprecision(s.decimals)<<*s.val;
                sf::Text valTxt(font,oss.str(),10);
                valTxt.setFillColor(sf::Color(100,220,180));
                valTxt.setPosition({PX+BAR_W+8,s.y-4}); win.draw(valTxt);

                sf::RectangleShape bar({BAR_W,BAR_H});
                bar.setFillColor(sf::Color(50,50,80));
                bar.setPosition({PX,s.y}); win.draw(bar);

                float t=(*s.val-s.min)/(s.max-s.min);
                sf::RectangleShape fill({BAR_W*t,BAR_H});
                fill.setFillColor(sf::Color(80,140,255));
                fill.setPosition({PX,s.y}); win.draw(fill);

                sf::CircleShape knob(6);
                knob.setFillColor(sf::Color(200,220,255));
                knob.setPosition({PX+t*BAR_W-6,s.y-4}); win.draw(knob);

                if (overPanel &&
                    sf::Mouse::isButtonPressed(sf::Mouse::Button::Left) &&
                    std::abs(float(m.y)-s.y)<16 && m.x>PX && m.x<PX+BAR_W+30)
                {
                    float nt=(m.x-PX)/BAR_W;
                    *s.val=s.min+std::clamp(nt,0.f,1.f)*(s.max-s.min);
                }
            }
        }

        win.display();
    }
}