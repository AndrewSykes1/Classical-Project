#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <iostream>

// ── resolution ────────────────────────────────────────────────────────────────
constexpr int WIN_W = 1400;
constexpr int WIN_H = 900;
constexpr int NX    = 280;
constexpr int NY    = 180;
constexpr float CELL = float(WIN_W) / NX;

constexpr int   ITER = 16;
constexpr float DT   = 0.08f;

float flowSpeed   = 2.5f;
float densityFade = 0.992f;
float viscosity   = 0.00001f;
float diffusion   = 0.00001f;
float brushRadius = 1.5f;

inline int IX(int x, int y) { return x + y * NX; }

std::vector<float> u(NX*NY),v(NX*NY),uPrev(NX*NY),vPrev(NX*NY);
std::vector<float> dens(NX*NY),densPrev(NX*NY);
std::vector<float> pressure(NX*NY),pdiv(NX*NY);
std::vector<bool>  solid(NX*NY,false);

enum class Mode { Density, Vorticity, Pressure };
Mode vizMode    = Mode::Density;
bool showStreams = false;
bool windTunnel = false;
float emaVort=1.f, emaPres=1.f;
constexpr float EMA_A=0.05f;

// ── solver ────────────────────────────────────────────────────────────────────
void enforce_solid_velocity() {
    for(int j=1;j<NY-1;j++) for(int i=1;i<NX-1;i++){
        if(solid[IX(i,j)]){u[IX(i,j)]=v[IX(i,j)]=0;continue;}
        if(solid[IX(i+1,j)]&&u[IX(i,j)]>0) u[IX(i,j)]=0;
        if(solid[IX(i-1,j)]&&u[IX(i,j)]<0) u[IX(i,j)]=0;
        if(solid[IX(i,j+1)]&&v[IX(i,j)]>0) v[IX(i,j)]=0;
        if(solid[IX(i,j-1)]&&v[IX(i,j)]<0) v[IX(i,j)]=0;
    }
}
void set_bnd(int b,std::vector<float>&x){
    for(int i=1;i<NX-1;i++){x[IX(i,0)]=b==2?-x[IX(i,1)]:x[IX(i,1)];x[IX(i,NY-1)]=b==2?-x[IX(i,NY-2)]:x[IX(i,NY-2)];}
    for(int j=1;j<NY-1;j++){x[IX(0,j)]=b==1?-x[IX(1,j)]:x[IX(1,j)];x[IX(NX-1,j)]=b==1?-x[IX(NX-2,j)]:x[IX(NX-2,j)];}
    for(int n=0;n<NX*NY;n++) if(solid[n]) x[n]=0;
}
void lin_solve(int b,std::vector<float>&x,std::vector<float>&x0,float a,float c){
    float ic=1.f/c;
    for(int k=0;k<ITER;k++){
        for(int j=1;j<NY-1;j++) for(int i=1;i<NX-1;i++){
            if(solid[IX(i,j)]){x[IX(i,j)]=0;continue;}
            x[IX(i,j)]=(x0[IX(i,j)]+a*(x[IX(i+1,j)]+x[IX(i-1,j)]+x[IX(i,j+1)]+x[IX(i,j-1)]))*ic;
        }
        set_bnd(b,x);
    }
}
void diffuse(int b,std::vector<float>&x,std::vector<float>&x0,float diff){
    float s=float(NX+NY)*0.5f,a=DT*diff*s*s;lin_solve(b,x,x0,a,1+4*a);
}
void project(std::vector<float>&pu,std::vector<float>&pv){
    float ia=2.f/float(NX+NY),ha=float(NX+NY)*0.5f;
    for(int j=1;j<NY-1;j++) for(int i=1;i<NX-1;i++){
        pdiv[IX(i,j)]=-0.5f*(pu[IX(i+1,j)]-pu[IX(i-1,j)]+pv[IX(i,j+1)]-pv[IX(i,j-1)])*ia;
        pressure[IX(i,j)]=0;
    }
    set_bnd(0,pdiv);set_bnd(0,pressure);lin_solve(0,pressure,pdiv,1,4);
    for(int j=1;j<NY-1;j++) for(int i=1;i<NX-1;i++){
        if(solid[IX(i,j)]){pu[IX(i,j)]=pv[IX(i,j)]=0;continue;}
        pu[IX(i,j)]-=0.5f*ha*(pressure[IX(i+1,j)]-pressure[IX(i-1,j)]);
        pv[IX(i,j)]-=0.5f*ha*(pressure[IX(i,j+1)]-pressure[IX(i,j-1)]);
    }
    set_bnd(1,pu);set_bnd(2,pv);enforce_solid_velocity();
}
void advect(int b,std::vector<float>&d,std::vector<float>&d0,std::vector<float>&au,std::vector<float>&av){
    float du=DT*float(NX),dv=DT*float(NY);
    for(int j=1;j<NY-1;j++) for(int i=1;i<NX-1;i++){
        if(solid[IX(i,j)]){d[IX(i,j)]=0;continue;}
        float x=std::clamp(i-du*au[IX(i,j)],0.5f,float(NX)-1.5f);
        float y=std::clamp(j-dv*av[IX(i,j)],0.5f,float(NY)-1.5f);
        int i0=int(x),i1=i0+1,j0=int(y),j1=j0+1;
        float s1=x-i0,s0=1-s1,t1=y-j0,t0=1-t1;
        auto s=[&](int a,int b){return solid[IX(a,b)]?0.f:d0[IX(a,b)];};
        d[IX(i,j)]=s0*(t0*s(i0,j0)+t1*s(i0,j1))+s1*(t0*s(i1,j0)+t1*s(i1,j1));
    }
    set_bnd(b,d);
}
void velocity_step(){
    diffuse(1,uPrev,u,viscosity);diffuse(2,vPrev,v,viscosity);
    enforce_solid_velocity();project(uPrev,vPrev);
    advect(1,u,uPrev,uPrev,vPrev);advect(2,v,vPrev,uPrev,vPrev);
    enforce_solid_velocity();project(u,v);
}
void density_step(){
    diffuse(0,densPrev,dens,diffusion);advect(0,dens,densPrev,u,v);
    for(auto&d:dens)d*=densityFade;
    for(int n=0;n<NX*NY;n++) if(solid[n]) dens[n]=0;
}
void reset_fluid(){
    std::fill(dens.begin(),dens.end(),0);std::fill(densPrev.begin(),densPrev.end(),0);
    std::fill(u.begin(),u.end(),0);std::fill(v.begin(),v.end(),0);
    std::fill(uPrev.begin(),uPrev.end(),0);std::fill(vPrev.begin(),vPrev.end(),0);
    std::fill(pressure.begin(),pressure.end(),0);std::fill(pdiv.begin(),pdiv.end(),0);
}
void paint(int gx,int gy,bool draw){
    int r=int(brushRadius);
    for(int dy=-r;dy<=r;dy++) for(int dx=-r;dx<=r;dx++){
        if(dx*dx+dy*dy>r*r) continue;
        int px=gx+dx,py=gy+dy;
        if(px>1&&px<NX-2&&py>1&&py<NY-2){
            solid[IX(px,py)]=draw;
            if(!draw){u[IX(px,py)]=v[IX(px,py)]=dens[IX(px,py)]=0;}
        }
    }
}

// ── colour maps ───────────────────────────────────────────────────────────────
sf::Color densityColor(float d){d=std::clamp(d,0.f,1.f);return{uint8_t(d*80),uint8_t(d*200),uint8_t(40+d*215)};}
sf::Color vorticityColor(float t){float m=std::pow(std::abs(t),0.5f);if(t>=0)return{uint8_t(255*m),uint8_t(30*(1-m)),uint8_t(80*(1-m))};return{uint8_t(80*(1-m)),uint8_t(30*(1-m)),uint8_t(255*m)};}
sf::Color pressureColor(float t){float m=std::pow(std::abs(t),0.6f);if(t>=0)return{uint8_t(255*m),uint8_t(120*m*(1-m*0.5f)),0};return{uint8_t(60*m),uint8_t(20*m),uint8_t(255*m)};}

// ── vehicle ───────────────────────────────────────────────────────────────────
struct CropUV{float u0=0,v0=0,u1=1,v1=1;};
struct Vehicle{std::string name,path;float defScale;bool invert;float aCut,bCut;CropUV crop;};

std::vector<Vehicle> vehicles={
    {"Land Cruiser","landcruiser_top.png",0.18f,false,30.f,0.5f,{0.f,0.35f,0.55f,0.75f}},
    {"Corvette",    "corvette_top.png",   0.35f,false,30.f,0.5f,{0.f,0.f,  1.f, 0.85f}},
};
int currentVehicle=-1;

struct PixCache{std::vector<std::pair<float,float>>pts;int sw=0,sh=0;};
PixCache pxc;

// Placement parameters (used by both stamp and the placement panel)
float placeScale=0.18f, placeAngle=0.f, placeCX=0.5f, placeCY=0.5f;

void stamp(){
    for(int n=0;n<NX*NY;n++) solid[n]=false;
    if(windTunnel) for(int i=0;i<NX;i++){solid[IX(i,1)]=solid[IX(i,NY-2)]=true;}
    float rad=placeAngle*3.14159265f/180.f,ca=std::cos(rad),sa=std::sin(rad);
    int cx=int(placeCX*NX), cy=int(placeCY*NY);
    for(auto&[lx,ly]:pxc.pts){
        float rx=lx*ca-ly*sa, ry=lx*sa+ly*ca;
        int gx=cx+int(rx*pxc.sw*placeScale);
        int gy=cy+int(ry*pxc.sh*placeScale);
        if(gx>=1&&gx<NX-1&&gy>=1&&gy<NY-1) solid[IX(gx,gy)]=true;
    }
    reset_fluid();
}

void load_vehicle(int idx){
    if(idx<0||idx>=(int)vehicles.size()) return;
    auto&vh=vehicles[idx];
    sf::Image img;
    if(!img.loadFromFile(vh.path)){std::cerr<<"cannot open "<<vh.path<<"\n";return;}
    auto[W,H]=img.getSize();
    unsigned px0=unsigned(vh.crop.u0*W),px1=unsigned(vh.crop.u1*W);
    unsigned py0=unsigned(vh.crop.v0*H),py1=unsigned(vh.crop.v1*H);
    if(px1<=px0)px1=W; if(py1<=py0)py1=H;
    unsigned cW=px1-px0,cH=py1-py0;
    pxc.pts.clear(); pxc.sw=int(cW); pxc.sh=int(cH);
    for(unsigned py=py0;py<py1;py++) for(unsigned px=px0;px<px1;px++){
        sf::Color c=img.getPixel({px,py});
        if(float(c.a)<vh.aCut) continue;
        float br=(float(c.r)+c.g+c.b)/(3.f*255.f);
        bool filled=vh.invert?(br>vh.bCut):(br<vh.bCut);
        if(!filled) continue;
        pxc.pts.push_back({float(px-px0)/float(cW)-0.5f, float(py-py0)/float(cH)-0.5f});
    }
    placeScale=vh.defScale; placeAngle=0.f; placeCX=0.5f; placeCY=0.5f;
    currentVehicle=idx;
    stamp();
    std::cout<<"loaded "<<vh.name<<" ("<<pxc.pts.size()<<" pts)\n";
}

// ── draggable panel ───────────────────────────────────────────────────────────
struct Panel{
    sf::Vector2f pos,size;
    bool drag=false;
    sf::Vector2f doff;
    bool hit(sf::Vector2i m) const{return m.x>=pos.x&&m.x<pos.x+size.x&&m.y>=pos.y&&m.y<pos.y+size.y;}
    bool titleHit(sf::Vector2i m) const{return m.x>=pos.x&&m.x<pos.x+size.x&&m.y>=pos.y&&m.y<pos.y+20;}
    void startDrag(sf::Vector2i m){drag=true;doff={float(m.x)-pos.x,float(m.y)-pos.y};}
    void moveDrag(sf::Vector2i m){if(drag)pos={float(m.x)-doff.x,float(m.y)-doff.y};}
    void stopDrag(){drag=false;}
    void drawBg(sf::RenderWindow&win,const sf::Font&font,const std::string&title) const{
        sf::RectangleShape bg(size); bg.setPosition(pos);
        bg.setFillColor(sf::Color(14,14,26,238));
        bg.setOutlineColor(sf::Color(65,65,105)); bg.setOutlineThickness(1);
        win.draw(bg);
        sf::RectangleShape tb({size.x,20}); tb.setPosition(pos);
        tb.setFillColor(sf::Color(30,38,78,245)); win.draw(tb);
        sf::Text t(font,title,11); t.setStyle(sf::Text::Bold);
        t.setFillColor(sf::Color(155,190,255)); t.setPosition({pos.x+6,pos.y+3});
        win.draw(t);
    }
};

// ── slider helper ─────────────────────────────────────────────────────────────
void drawSlider(sf::RenderWindow&win,const sf::Font&font,
                float*val,float mn,float mx,const char*label,int dec,
                float px,float py,float pw){
    float t=(*val-mn)/(mx-mn);
    sf::Text lb(font,label,10); lb.setFillColor(sf::Color(185,208,255));
    lb.setPosition({px,py-16}); win.draw(lb);
    std::ostringstream os; os<<std::fixed<<std::setprecision(dec)<<*val;
    sf::Text vt(font,os.str(),10); vt.setFillColor(sf::Color(100,220,180));
    vt.setPosition({px+pw+6,py-4}); win.draw(vt);
    sf::RectangleShape bar({pw,4}); bar.setFillColor(sf::Color(42,42,72));
    bar.setPosition({px,py}); win.draw(bar);
    sf::RectangleShape fill({pw*t,4}); fill.setFillColor(sf::Color(75,135,255));
    fill.setPosition({px,py}); win.draw(fill);
    sf::CircleShape knob(5); knob.setFillColor(sf::Color(210,225,255));
    knob.setPosition({px+t*pw-5,py-3}); win.draw(knob);
}
bool interactSlider(float*val,float mn,float mx,float px,float py,float pw,
                    sf::Vector2i m,bool ld){
    if(!ld||m.x<px||m.x>px+pw||std::abs(float(m.y)-py)>14) return false;
    *val=mn+std::clamp((float(m.x)-px)/pw,0.f,1.f)*(mx-mn);
    return true;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(){
    sf::RenderWindow win(sf::VideoMode({WIN_W,WIN_H}),"Airflow Simulator");
    win.setFramerateLimit(60);
    sf::Font font;
    if(!font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) return 1;

    Panel cp; cp.pos={10,10};  cp.size={252,400};   // controls
    Panel pp; pp.pos={272,10}; pp.size={258,320};   // placement

    bool showCtrl=true, showPlace=false;

    sf::VertexArray grid(sf::PrimitiveType::Triangles,NX*NY*6);

    sf::Text hint(font,"P=controls  O=placement  S=streams  V=vorticity  Q=pressure  W=tunnel  C=clear  1/2=vehicle",11);
    hint.setFillColor(sf::Color(165,165,165,200));
    hint.setPosition({8.f,float(WIN_H)-20.f});

    load_vehicle(0); showPlace=true;

    bool lWas=false;

    while(win.isOpen()){
        sf::Vector2i m=sf::Mouse::getPosition(win);
        bool ld=sf::Mouse::isButtonPressed(sf::Mouse::Button::Left);
        bool lclick=ld&&!lWas;

        while(auto e=win.pollEvent()){
            if(e->is<sf::Event::Closed>()) win.close();
            if(auto*k=e->getIf<sf::Event::KeyPressed>()){
                switch(k->code){
                case sf::Keyboard::Key::P: showCtrl=!showCtrl; break;
                case sf::Keyboard::Key::O: showPlace=!showPlace; break;
                case sf::Keyboard::Key::S: showStreams=!showStreams; break;
                case sf::Keyboard::Key::V: vizMode=(vizMode==Mode::Vorticity)?Mode::Density:Mode::Vorticity; break;
                case sf::Keyboard::Key::Q: vizMode=(vizMode==Mode::Pressure)?Mode::Density:Mode::Pressure; break;
                case sf::Keyboard::Key::W:
                    windTunnel=!windTunnel;
                    for(int i=0;i<NX;i++){solid[IX(i,1)]=windTunnel;solid[IX(i,NY-2)]=windTunnel;}
                    break;
                case sf::Keyboard::Key::C:
                    std::fill(solid.begin(),solid.end(),false);
                    if(windTunnel) for(int i=0;i<NX;i++) solid[IX(i,1)]=solid[IX(i,NY-2)]=true;
                    reset_fluid(); currentVehicle=-1; pxc.pts.clear(); break;
                case sf::Keyboard::Key::Num1: load_vehicle(0); showPlace=true; break;
                case sf::Keyboard::Key::Num2: load_vehicle(1); showPlace=true; break;
                default: break;
                }
            }
            if(auto*mp=e->getIf<sf::Event::MouseButtonPressed>()){
                if(mp->button==sf::Mouse::Button::Left){
                    if(showCtrl&&cp.titleHit(m))  cp.startDrag(m);
                    if(showPlace&&pp.titleHit(m)) pp.startDrag(m);
                }
            }
            if(auto*mr=e->getIf<sf::Event::MouseButtonReleased>()){
                if(mr->button==sf::Mouse::Button::Left){cp.stopDrag();pp.stopDrag();}
            }
        }
        cp.moveDrag(m); pp.moveDrag(m);

        bool overC=showCtrl&&cp.hit(m);
        bool overP=showPlace&&pp.hit(m);
        bool overAny=overC||overP;

        // ── placement sliders ─────────────────────────────────────────────────
        bool needStamp=false;
        if(showPlace&&currentVehicle>=0&&!pp.drag){
            float sx=pp.pos.x+12, sw=pp.size.x-58, rowH=52.f;
            struct PS{float*v;float mn,mx;const char*l;int d;};
            PS ps[]={
                {&placeScale, 0.05f,1.0f, "Scale",   3},
                {&placeAngle,-180.f,180.f,"Rotation",1},
                {&placeCX,   0.f,  1.f,  "Pos X",   2},
                {&placeCY,   0.f,  1.f,  "Pos Y",   2},
            };
            for(int i=0;i<4;i++){
                float py=pp.pos.y+32+i*rowH+18;
                if(interactSlider(ps[i].v,ps[i].mn,ps[i].mx,sx,py,sw,m,ld)) needStamp=true;
            }
            if(needStamp) stamp();

            // Apply button
            float bx=pp.pos.x+12, by=pp.pos.y+pp.size.y-38;
            float bw=pp.size.x-24;
            if(lclick&&m.x>=bx&&m.x<bx+bw&&m.y>=by&&m.y<by+24) stamp();

            // vehicle swap buttons inside placement panel
            for(int vi=0;vi<(int)vehicles.size();vi++){
                float vbx=pp.pos.x+12, vby=pp.pos.y+pp.size.y-70-(vi*24.f);
                float vbw=pp.size.x-24;
                if(lclick&&m.x>=vbx&&m.x<vbx+vbw&&m.y>=vby&&m.y<vby+20){
                    load_vehicle(vi);
                }
            }
        }

        // ── control sliders ───────────────────────────────────────────────────
        if(showCtrl&&!cp.drag){
            float sx=cp.pos.x+12, sw=cp.size.x-58, rowH=52.f;
            struct CS{float*v;float mn,mx;const char*l;int d;};
            CS cs[]={
                {&flowSpeed,   0.5f,  8.f,    "Flow Speed",   2},
                {&densityFade, 0.95f, 0.999f, "Smoke Persist",3},
                {&viscosity,   0.f,   0.001f, "Viscosity",    5},
                {&diffusion,   0.f,   0.001f, "Diffusion",    5},
                {&brushRadius, 0.5f,  8.f,    "Brush Size",   1},
            };
            for(int i=0;i<5;i++){
                float py=cp.pos.y+110+i*rowH;
                interactSlider(cs[i].v,cs[i].mn,cs[i].mx,sx,py,sw,m,ld);
            }
            // vehicle buttons in ctrl panel
            for(int vi=0;vi<(int)vehicles.size();vi++){
                float bx=cp.pos.x+12, by=cp.pos.y+50+vi*22.f, bw=cp.size.x-24;
                if(lclick&&m.x>=bx&&m.x<bx+bw&&m.y>=by&&m.y<by+18){
                    load_vehicle(vi); showPlace=true;
                }
            }
        }

        // ── paint ─────────────────────────────────────────────────────────────
        if(!overAny){
            int gx=int(m.x/CELL),gy=int(m.y/CELL);
            if(ld) paint(gx,gy,true);
            if(sf::Mouse::isButtonPressed(sf::Mouse::Button::Right)) paint(gx,gy,false);
        }

        for(int j=1;j<NY-1;j++){u[IX(1,j)]=flowSpeed;dens[IX(1,j)]=1.f;}
        velocity_step(); density_step();

        if(vizMode==Mode::Vorticity){
            float fm=1e-6f;
            for(int j=1;j<NY-1;j++) for(int i=1;i<NX-1;i++){
                float w=(v[IX(i+1,j)]-v[IX(i-1,j)]-u[IX(i,j+1)]+u[IX(i,j-1)])*0.5f;
                fm=std::max(fm,std::abs(w));
            }
            emaVort=emaVort*(1-EMA_A)+fm*EMA_A;
        }
        if(vizMode==Mode::Pressure){
            float fm=1e-6f;
            for(auto p:pressure) fm=std::max(fm,std::abs(p));
            emaPres=emaPres*(1-EMA_A)+fm*EMA_A;
        }

        for(int j=0;j<NY;j++) for(int i=0;i<NX;i++){
            float x0=i*CELL,y0=j*CELL,x1=x0+CELL+1,y1=y0+CELL+1;
            sf::Color col;
            if(solid[IX(i,j)]) col=sf::Color(220,60,60);
            else switch(vizMode){
            case Mode::Density:   col=densityColor(dens[IX(i,j)]); break;
            case Mode::Vorticity:{
                float w=0;
                if(i>0&&i<NX-1&&j>0&&j<NY-1)
                    w=(v[IX(i+1,j)]-v[IX(i-1,j)]-u[IX(i,j+1)]+u[IX(i,j-1)])*0.5f;
                col=vorticityColor(w/emaVort); break;
            }
            case Mode::Pressure: col=pressureColor(pressure[IX(i,j)]/emaPres); break;
            }
            int base=(j*NX+i)*6;
            sf::Vertex tl{{x0,y0},col},tr{{x1,y0},col},bl{{x0,y1},col},br{{x1,y1},col};
            grid[base+0]=tl;grid[base+1]=tr;grid[base+2]=br;
            grid[base+3]=tl;grid[base+4]=br;grid[base+5]=bl;
        }

        win.clear(sf::Color(10,10,20));
        win.draw(grid);

        if(showStreams){
            constexpr int STEP=14; constexpr float AL=4.f;
            sf::VertexArray lines(sf::PrimitiveType::Lines);
            for(int j=STEP/2;j<NY;j+=STEP) for(int i=STEP/2;i<NX;i+=STEP){
                if(solid[IX(i,j)]) continue;
                float uu=u[IX(i,j)],vv=v[IX(i,j)];
                float sp=std::sqrt(uu*uu+vv*vv); if(sp<0.01f) continue;
                float nx=uu/sp,ny=vv/sp;
                float cx=(i+.5f)*CELL,cy=(j+.5f)*CELL,ex=cx+nx*AL*2,ey=cy+ny*AL*2;
                uint8_t br=uint8_t(std::min(255.f,sp/flowSpeed*220));
                sf::Color ac(br,br,255,190);
                lines.append({{cx,cy},ac});lines.append({{ex,ey},ac});
                float lx=nx*AL-ny*AL*.5f,ly=ny*AL+nx*AL*.5f;
                lines.append({{ex,ey},ac});lines.append({{ex-lx,ey-ly},ac});
                float rx=nx*AL+ny*AL*.5f,ry=ny*AL-nx*AL*.5f;
                lines.append({{ex,ey},ac});lines.append({{ex-rx,ey-ry},ac});
            }
            win.draw(lines);
        }

        win.draw(hint);

        // ── draw controls panel ───────────────────────────────────────────────
        if(showCtrl){
            cp.drawBg(win,font,"CONTROLS  \u2261 drag to move");
            float px=cp.pos.x,py=cp.pos.y;
            float sx=px+12,sw=cp.size.x-58,rowH=52.f;

            const char*ms=vizMode==Mode::Vorticity?"VORTICITY":vizMode==Mode::Pressure?"PRESSURE":"DENSITY";
            sf::Text ml(font,std::string("view: ")+ms,10);
            ml.setFillColor(sf::Color(100,220,150));ml.setPosition({px+12,py+24});win.draw(ml);
            sf::Text wl(font,windTunnel?"tunnel:ON":"tunnel:OFF",10);
            wl.setFillColor(windTunnel?sf::Color(100,220,100):sf::Color(130,130,155));
            wl.setPosition({px+148,py+24});win.draw(wl);

            for(int vi=0;vi<(int)vehicles.size();vi++){
                bool act=(currentVehicle==vi);
                float bx=px+12,by=py+40+vi*22.f,bw=cp.size.x-24;
                sf::RectangleShape btn({bw,18});btn.setPosition({bx,by});
                btn.setFillColor(act?sf::Color(48,78,158,225):sf::Color(26,26,46,210));
                btn.setOutlineColor(act?sf::Color(100,160,255):sf::Color(52,52,82));
                btn.setOutlineThickness(1);win.draw(btn);
                sf::Text vt(font,std::to_string(vi+1)+": "+vehicles[vi].name,10);
                vt.setFillColor(act?sf::Color(200,230,255):sf::Color(150,162,182));
                vt.setPosition({bx+4,by+2});win.draw(vt);
            }

            struct CS{float*v;float mn,mx;const char*l;int d;};
            CS cs[]={
                {&flowSpeed,   0.5f,  8.f,    "Flow Speed",   2},
                {&densityFade, 0.95f, 0.999f, "Smoke Persist",3},
                {&viscosity,   0.f,   0.001f, "Viscosity",    5},
                {&diffusion,   0.f,   0.001f, "Diffusion",    5},
                {&brushRadius, 0.5f,  8.f,    "Brush Size",   1},
            };
            for(int i=0;i<5;i++){
                float sy=py+110+i*rowH;
                drawSlider(win,font,cs[i].v,cs[i].mn,cs[i].mx,cs[i].l,cs[i].d,sx,sy,sw);
            }
        }

        // ── draw placement panel ──────────────────────────────────────────────
        if(showPlace){
            pp.drawBg(win,font,"PLACEMENT  \u2261 drag to move");
            float px=pp.pos.x,py=pp.pos.y;
            float sx=px+12,sw=pp.size.x-58,rowH=52.f;

            if(currentVehicle<0){
                sf::Text nt(font,"Load a vehicle first (press 1 or 2)",10);
                nt.setFillColor(sf::Color(180,140,80));
                nt.setPosition({px+10,py+28});win.draw(nt);
            } else {
                struct PS{float*v;float mn,mx;const char*l;int d;};
                PS ps[]={
                    {&placeScale, 0.05f,1.f,  "Scale",   3},
                    {&placeAngle,-180.f,180.f,"Rotation",1},
                    {&placeCX,   0.f,  1.f,  "Pos X",   2},
                    {&placeCY,   0.f,  1.f,  "Pos Y",   2},
                };
                for(int i=0;i<4;i++){
                    float sy=py+32+i*rowH+18;
                    drawSlider(win,font,ps[i].v,ps[i].mn,ps[i].mx,ps[i].l,ps[i].d,sx,sy,sw);
                }

                // vehicle swap buttons
                for(int vi=0;vi<(int)vehicles.size();vi++){
                    bool act=(currentVehicle==vi);
                    float bx=px+12, by=py+pp.size.y-74-vi*24.f, bw=pp.size.x-24;
                    sf::RectangleShape btn({bw,20});btn.setPosition({bx,by});
                    btn.setFillColor(act?sf::Color(48,78,158,225):sf::Color(26,26,46,210));
                    btn.setOutlineColor(act?sf::Color(100,160,255):sf::Color(52,52,82));
                    btn.setOutlineThickness(1);win.draw(btn);
                    sf::Text vt(font,std::to_string(vi+1)+": "+vehicles[vi].name,10);
                    vt.setFillColor(act?sf::Color(200,230,255):sf::Color(150,162,182));
                    vt.setPosition({bx+4,by+3});win.draw(vt);
                }

                // Apply / reset-flow button
                float bx=px+12,by=py+pp.size.y-38,bw=pp.size.x-24;
                sf::RectangleShape ab({bw,24});ab.setPosition({bx,by});
                ab.setFillColor(sf::Color(38,95,55,230));
                ab.setOutlineColor(sf::Color(75,195,115));ab.setOutlineThickness(1);
                win.draw(ab);
                sf::Text at(font,"Apply / Reset Flow",11);
                at.setFillColor(sf::Color(135,240,165));at.setPosition({bx+10,by+4});win.draw(at);

                // live readout
                std::ostringstream info;
                info<<"cx="<<int(placeCX*NX)<<" cy="<<int(placeCY*NY)
                    <<"  scale="<<std::fixed<<std::setprecision(3)<<placeScale
                    <<"  rot="<<std::fixed<<std::setprecision(1)<<placeAngle<<"deg";
                sf::Text ir(font,info.str(),9);
                ir.setFillColor(sf::Color(110,110,150));
                ir.setPosition({px+10,py+pp.size.y-52});win.draw(ir);
            }
        }

        lWas=ld;
        win.display();
    }
}