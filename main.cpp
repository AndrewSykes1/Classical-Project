#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

// ── tunables ──────────────────────────────────────────────────────────────────
constexpr int WIN_W = 1400;
constexpr int WIN_H = 900;
constexpr int N     = 256;   // higher grid resolution (was 180)
constexpr int ITER  = 14;    // slightly fewer iterations to compensate

float flowSpeed   = 2.5f;
float densityFade = 0.992f;
float viscosity   = 0.00001f;
float diffusion   = 0.00001f;
float brushRadius = 1.5f;

constexpr float DT = 0.08f;

float SCALE_X = float(WIN_W) / N;
float SCALE_Y = float(WIN_H) / N;

inline int IX(int x, int y) { return x + y * N; }

std::vector<float> u(N*N), v(N*N), uPrev(N*N), vPrev(N*N);
std::vector<float> dens(N*N), densPrev(N*N);
std::vector<bool>  solid(N*N, false);

// ── display modes ─────────────────────────────────────────────────────────────
enum class Mode { Density, Vorticity, Pressure };
Mode    vizMode      = Mode::Density;
bool    showStreams   = false;
bool    windTunnel   = false;

// ── boundary ──────────────────────────────────────────────────────────────────
void set_bnd(int b, std::vector<float>& x) {
    for (int i = 1; i < N-1; i++) {
        x[IX(0,   i)] = b==1 ? -x[IX(1,   i)] : x[IX(1,   i)];
        x[IX(N-1, i)] = b==1 ? -x[IX(N-2, i)] : x[IX(N-2, i)];
        x[IX(i,   0)] = b==2 ? -x[IX(i,   1)] : x[IX(i,   1)];
        x[IX(i, N-1)] = b==2 ? -x[IX(i, N-2)] : x[IX(i, N-2)];
    }
    for (int j = 1; j < N-1; j++) {
        for (int i = 1; i < N-1; i++) {
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

void lin_solve(int b, std::vector<float>& x, std::vector<float>& x0,
               float a, float c) {
    for (int k = 0; k < ITER; k++) {
        for (int j = 1; j < N-1; j++)
            for (int i = 1; i < N-1; i++) {
                if (solid[IX(i,j)]) { x[IX(i,j)] = 0; continue; }
                x[IX(i,j)] = (x0[IX(i,j)] + a*(
                    x[IX(i+1,j)] + x[IX(i-1,j)] +
                    x[IX(i,j+1)] + x[IX(i,j-1)]
                )) / c;
            }
        set_bnd(b, x);
    }
}

void diffuse(int b, std::vector<float>& x, std::vector<float>& x0, float diff) {
    float a = DT * diff * (N-2) * (N-2);
    lin_solve(b, x, x0, a, 1+4*a);
}

void project(std::vector<float>& pu, std::vector<float>& pv,
             std::vector<float>& p,  std::vector<float>& div) {
    for (int j = 1; j < N-1; j++)
        for (int i = 1; i < N-1; i++) {
            div[IX(i,j)] = -0.5f*(
                pu[IX(i+1,j)] - pu[IX(i-1,j)] +
                pv[IX(i,j+1)] - pv[IX(i,j-1)]
            ) / N;
            p[IX(i,j)] = 0;
        }
    set_bnd(0, div); set_bnd(0, p);
    lin_solve(0, p, div, 1, 4);

    for (int j = 1; j < N-1; j++)
        for (int i = 1; i < N-1; i++) {
            if (solid[IX(i,j)]) { pu[IX(i,j)] = pv[IX(i,j)] = 0; continue; }
            pu[IX(i,j)] -= 0.5f*N*(p[IX(i+1,j)] - p[IX(i-1,j)]);
            pv[IX(i,j)] -= 0.5f*N*(p[IX(i,j+1)] - p[IX(i,j-1)]);
        }
    set_bnd(1, pu); set_bnd(2, pv);
}

void advect(int b, std::vector<float>& d, std::vector<float>& d0,
            std::vector<float>& au, std::vector<float>& av) {
    float dt0 = DT * N;
    for (int j = 1; j < N-1; j++)
        for (int i = 1; i < N-1; i++) {
            if (solid[IX(i,j)]) { d[IX(i,j)] = 0; continue; }
            float x = i - dt0 * au[IX(i,j)];
            float y = j - dt0 * av[IX(i,j)];
            x = std::clamp(x, 0.5f, float(N)-1.5f);
            y = std::clamp(y, 0.5f, float(N)-1.5f);
            int i0=int(x), i1=i0+1, j0=int(y), j1=j0+1;
            float s1=x-i0, s0=1-s1, t1=y-j0, t0=1-t1;
            auto samp = [&](int si, int sj) {
                return solid[IX(si,sj)] ? 0.f : d0[IX(si,sj)];
            };
            d[IX(i,j)] =
                s0*(t0*samp(i0,j0)+t1*samp(i0,j1)) +
                s1*(t0*samp(i1,j0)+t1*samp(i1,j1));
        }
    set_bnd(b, d);
}

void velocity_step() {
    diffuse(1, uPrev, u, viscosity);
    diffuse(2, vPrev, v, viscosity);
    project(uPrev, vPrev, u, v);
    advect(1, u, uPrev, uPrev, vPrev);
    advect(2, v, vPrev, uPrev, vPrev);
    project(u, v, uPrev, vPrev);
}

void density_step() {
    diffuse(0, densPrev, dens, diffusion);
    advect(0, dens, densPrev, u, v);
    for (auto& d : dens) d *= densityFade;
}

void paint(int gx, int gy, bool draw) {
    int r = int(brushRadius);
    for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++) {
            if (dx*dx+dy*dy > r*r) continue;
            int px=gx+dx, py=gy+dy;
            if (px>1 && px<N-2 && py>1 && py<N-2)
                solid[IX(px,py)] = draw;
        }
}

// ── colour helpers ────────────────────────────────────────────────────────────
sf::Color vorticityColor(float w, float maxW) {
    float t = std::clamp(w / (maxW + 1e-6f), -1.f, 1.f);
    if (t >= 0) {
        return {uint8_t(255*t), uint8_t(20*(1-t)), uint8_t(20*(1-t))};
    } else {
        t = -t;
        return {uint8_t(20*(1-t)), uint8_t(20*(1-t)), uint8_t(255*t)};
    }
}

sf::Color densityColor(float d) {
    d = std::clamp(d, 0.f, 1.f);
    return {uint8_t(d*80), uint8_t(d*200), uint8_t(40+d*215)};
}

sf::Color pressureColor(float p, float maxP) {
    float t = std::clamp(0.5f + p/(2.f*maxP + 1e-6f), 0.f, 1.f);
    float r, g, b;
    if      (t < 0.25f) { float s=t/0.25f;        r=0;   g=s;   b=1;   }
    else if (t < 0.5f)  { float s=(t-0.25f)/0.25f; r=0;   g=1;   b=1-s; }
    else if (t < 0.75f) { float s=(t-0.5f)/0.25f;  r=s;   g=1;   b=0;   }
    else                { float s=(t-0.75f)/0.25f;  r=1;   g=1-s; b=0;   }
    return {uint8_t(r*255), uint8_t(g*255), uint8_t(b*255)};
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

    sf::VertexArray grid(sf::PrimitiveType::Triangles, N*N*6);
    std::vector<float> pressure(N*N, 0), pdiv(N*N, 0);

    sf::Text hint(font,
        "P=panel  S=streamlines  V=vorticity  Q=pressure  W=wind tunnel  C=clear",
        11);
    hint.setFillColor(sf::Color(180,180,180,200));
    hint.setPosition({8.f, float(WIN_H)-20.f});

    while (win.isOpen()) {

        while (auto e = win.pollEvent()) {
            if (e->is<sf::Event::Closed>()) win.close();
            if (auto* k = e->getIf<sf::Event::KeyPressed>()) {
                switch (k->code) {
                case sf::Keyboard::Key::P: showPanel  = !showPanel;  break;
                case sf::Keyboard::Key::S: showStreams = !showStreams; break;
                case sf::Keyboard::Key::V:
                    vizMode = (vizMode==Mode::Vorticity) ? Mode::Density : Mode::Vorticity;
                    break;
                case sf::Keyboard::Key::Q:
                    vizMode = (vizMode==Mode::Pressure) ? Mode::Density : Mode::Pressure;
                    break;
                case sf::Keyboard::Key::W:
                    windTunnel = !windTunnel;
                    for (int i=0; i<N; i++) {
                        solid[IX(i,1)]   = windTunnel;
                        solid[IX(i,N-2)] = windTunnel;
                    }
                    break;
                case sf::Keyboard::Key::C:
                    std::fill(solid.begin(), solid.end(), false);
                    if (windTunnel) for (int i=0;i<N;i++){
                        solid[IX(i,1)]=solid[IX(i,N-2)]=true; }
                    std::fill(dens.begin(), dens.end(), 0);
                    std::fill(u.begin(),    u.end(),    0);
                    std::fill(v.begin(),    v.end(),    0);
                    break;
                default: break;
                }
            }
        }

        auto m  = sf::Mouse::getPosition(win);
        int  gx = int(m.x / SCALE_X);
        int  gy = int(m.y / SCALE_Y);

        bool overPanel = showPanel && m.x < 240 && m.y < 330;
        if (!overPanel) {
            if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left))  paint(gx,gy,true);
            if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Right)) paint(gx,gy,false);
        }

        for (int j=1; j<N-1; j++) {
            u[IX(1,j)]    = flowSpeed;
            dens[IX(1,j)] = 1.f;
        }

        velocity_step();
        density_step();

        // pressure solve for pressure view
        if (vizMode == Mode::Pressure) {
            for (int j=1;j<N-1;j++) for (int i=1;i<N-1;i++) {
                pdiv[IX(i,j)] = -0.5f*(
                    u[IX(i+1,j)]-u[IX(i-1,j)] +
                    v[IX(i,j+1)]-v[IX(i,j-1)])/N;
                pressure[IX(i,j)]=0;
            }
            set_bnd(0,pdiv); set_bnd(0,pressure);
            lin_solve(0,pressure,pdiv,1,4);
        }

        // colour scaling passes
        float maxVort=1e-6f, maxPres=1e-6f;
        if (vizMode==Mode::Vorticity) {
            for (int j=1;j<N-1;j++) for (int i=1;i<N-1;i++) {
                float w=(v[IX(i+1,j)]-v[IX(i-1,j)]-(u[IX(i,j+1)]-u[IX(i,j-1)]))*0.5f;
                maxVort=std::max(maxVort,std::abs(w));
            }
        }
        if (vizMode==Mode::Pressure) {
            for (auto p : pressure) maxPres=std::max(maxPres,std::abs(p));
        }

        // build vertex array
        for (int j=0; j<N; j++) {
            for (int i=0; i<N; i++) {
                float x0=i*SCALE_X, y0=j*SCALE_Y;
                float x1=x0+SCALE_X+1, y1=y0+SCALE_Y+1;

                sf::Color col;
                if (solid[IX(i,j)]) {
                    col = sf::Color(220,60,60);
                } else {
                    switch (vizMode) {
                    case Mode::Density:   col = densityColor(dens[IX(i,j)]); break;
                    case Mode::Vorticity: {
                        float w=0;
                        if (i>0&&i<N-1&&j>0&&j<N-1)
                            w=(v[IX(i+1,j)]-v[IX(i-1,j)]-(u[IX(i,j+1)]-u[IX(i,j-1)]))*0.5f;
                        col = vorticityColor(w,maxVort);
                        break;
                    }
                    case Mode::Pressure:
                        col = pressureColor(pressure[IX(i,j)],maxPres);
                        break;
                    }
                }

                int base=(j*N+i)*6;
                sf::Vertex tl{{x0,y0},col}, tr{{x1,y0},col};
                sf::Vertex bl{{x0,y1},col}, br{{x1,y1},col};
                grid[base+0]=tl; grid[base+1]=tr; grid[base+2]=br;
                grid[base+3]=tl; grid[base+4]=br; grid[base+5]=bl;
            }
        }

        win.clear(sf::Color(10,10,20));
        win.draw(grid);

        // streamline arrows
        if (showStreams) {
            constexpr int  STEP  = 14;
            constexpr float ALEN = 5.f;
            sf::VertexArray lines(sf::PrimitiveType::Lines);

            for (int j=STEP/2; j<N; j+=STEP) {
                for (int i=STEP/2; i<N; i+=STEP) {
                    if (solid[IX(i,j)]) continue;
                    float uu=u[IX(i,j)], vv=v[IX(i,j)];
                    float spd=std::sqrt(uu*uu+vv*vv);
                    if (spd < 0.01f) continue;
                    float nx=uu/spd, ny=vv/spd;
                    float cx=(i+0.5f)*SCALE_X, cy=(j+0.5f)*SCALE_Y;
                    float ex=cx+nx*ALEN*2, ey=cy+ny*ALEN*2;
                    uint8_t br=uint8_t(std::min(255.f,spd/flowSpeed*200));
                    sf::Color ac(br,br,255,180);
                    lines.append({{cx,cy},ac}); lines.append({{ex,ey},ac});
                    float lx=nx*ALEN-ny*ALEN*0.5f, ly=ny*ALEN+nx*ALEN*0.5f;
                    lines.append({{ex,ey},ac}); lines.append({{ex-lx,ey-ly},ac});
                    float rx=nx*ALEN+ny*ALEN*0.5f, ry=ny*ALEN-nx*ALEN*0.5f;
                    lines.append({{ex,ey},ac}); lines.append({{ex-rx,ey-ry},ac});
                }
            }
            win.draw(lines);
        }

        win.draw(hint);

        // control panel
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

            sf::Text wtLabel(font, windTunnel?"tunnel: ON":"tunnel: OFF", 10);
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
                    nt=std::clamp(nt,0.f,1.f);
                    *s.val=s.min+nt*(s.max-s.min);
                }
            }
        }

        win.display();
    }
}