"""BISINDO — antarmuka kamera & penyusunan teks (v2.3).

Prediksi Kata Cerdas masih berupa placeholder visual nonaktif.
Pengenalan gestur, stabilisasi, dan penyusunan teks tetap aktif.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk

from PIL import Image, ImageDraw, ImageOps, ImageTk

from .camera import CameraWorker
from .core import AutoConfirmGate, TextComposer
from .word_prediction import suggest_words

HERE = Path(__file__).resolve().parents[2]
BG = "#EAF5FF"
WHITE = "#FFFFFF"
NAVY = "#0D3569"
BLUE = "#0768C2"
MUTED = "#526E94"
BORDER = "#BEDDF9"
TEAL = "#02A99C"
DARK_TEAL = "#008C7B"
PALE_BLUE = "#DDEEFF"
PALE_MINT = "#DDF8EF"
PALE_GOLD = "#FFF0CA"
PALE_PINK = "#FDE1E8"
FONT = "Segoe UI"


def rounded_polygon(canvas, xy, radius, **kwargs):
    """Smooth filled rectangle on any Tkinter Canvas."""
    x0, y0, x1, y1 = xy
    radius = max(1, min(radius, (x1-x0)/2, (y1-y0)/2))
    p = [x0+radius,y0, x1-radius,y0, x1,y0, x1,y0+radius,
         x1,y1-radius, x1,y1, x1-radius,y1, x0+radius,y1,
         x0,y1, x0,y1-radius, x0,y0+radius, x0,y0]
    return canvas.create_polygon(p, smooth=True, splinesteps=22, **kwargs)


def label(parent, value, size=11, color=NAVY, bold=False, bg=WHITE, **kwargs):
    return tk.Label(parent, text=value, fg=color, bg=bg,
                    font=(FONT, size, "bold" if bold else "normal"), **kwargs)


def draw_icon(canvas, name, cx, cy, color, scale=1):
    """Small vector icons, drawn without any extra icon packages."""
    t = max(1.8, 2.1*scale)
    kw = {"fill": color, "width": t, "capstyle": "round", "joinstyle": "round"}
    if name in ("camera", "video"):
        rounded_polygon(canvas, (cx-10*scale,cy-7*scale,cx+5*scale,cy+7*scale),
                        3*scale, fill="", outline=color, width=t)
        canvas.create_polygon(cx+5*scale,cy-4*scale, cx+12*scale,cy-8*scale,
                              cx+12*scale,cy+8*scale, cx+5*scale,cy+4*scale,
                              fill="", outline=color, width=t)
    elif name == "photo":
        canvas.create_rectangle(cx-10*scale,cy-9*scale,cx+10*scale,cy+9*scale,
                                fill="", outline=color, width=t)
        canvas.create_oval(cx+2*scale,cy-5*scale,cx+6*scale,cy-1*scale,fill=color,outline="")
    elif name == "bars":
        for dx, ht in [(-8,8),(-1,14),(6,20)]:
            rounded_polygon(canvas, (cx+dx*scale,cy+(10-ht)*scale,
                                     cx+(dx+4)*scale,cy+10*scale), 1.7*scale,
                            outline=color, fill="", width=t)
    elif name == "document":
        canvas.create_rectangle(cx-9*scale,cy-11*scale,cx+8*scale,cy+11*scale,
                                fill="", outline=color, width=t)
        canvas.create_line(cx-5*scale,cy-4*scale,cx+4*scale,cy-4*scale,**kw)
        canvas.create_line(cx-5*scale,cy+1*scale,cx+4*scale,cy+1*scale,**kw)
        canvas.create_line(cx-5*scale,cy+6*scale,cx+2*scale,cy+6*scale,**kw)
    elif name == "sparkles":
        canvas.create_line(cx-4*scale,cy-10*scale,cx-4*scale,cy+5*scale,**kw)
        canvas.create_line(cx-11*scale,cy-3*scale,cx+3*scale,cy-3*scale,**kw)
        canvas.create_line(cx+6*scale,cy+1*scale,cx+6*scale,cy+11*scale,**kw)
        canvas.create_line(cx+1*scale,cy+6*scale,cx+11*scale,cy+6*scale,**kw)
        canvas.create_line(cx+7*scale,cy-10*scale,cx+7*scale,cy-5*scale,**kw)
        canvas.create_line(cx+5*scale,cy-8*scale,cx+9*scale,cy-8*scale,**kw)
    elif name == "backspace":
        pts = [cx-10*scale,cy,cx-4*scale,cy-7*scale,cx+11*scale,cy-7*scale,
               cx+11*scale,cy+7*scale,cx-4*scale,cy+7*scale]
        canvas.create_polygon(pts,fill="",outline=color,width=t,joinstyle="round")
        canvas.create_line(cx+1*scale,cy-3*scale,cx+7*scale,cy+3*scale,**kw)
        canvas.create_line(cx+7*scale,cy-3*scale,cx+1*scale,cy+3*scale,**kw)
    elif name == "space":
        canvas.create_line(cx-11*scale,cy-5*scale,cx-11*scale,cy+6*scale,
                           cx+11*scale,cy+6*scale,cx+11*scale,cy-5*scale,**kw)
    elif name == "trash":
        canvas.create_rectangle(cx-7*scale,cy-5*scale,cx+7*scale,cy+9*scale,
                                fill="",outline=color,width=t)
        canvas.create_line(cx-10*scale,cy-8*scale,cx+10*scale,cy-8*scale,**kw)
        canvas.create_line(cx-3*scale,cy-11*scale,cx+3*scale,cy-11*scale,**kw)
        canvas.create_line(cx-2*scale,cy-2*scale,cx-2*scale,cy+5*scale,**kw)
        canvas.create_line(cx+2*scale,cy-2*scale,cx+2*scale,cy+5*scale,**kw)
    elif name == "check":
        canvas.create_oval(cx-11*scale,cy-11*scale,cx+11*scale,cy+11*scale,
                           fill=WHITE,outline="")
        canvas.create_line(cx-6*scale,cy,cx-2*scale,cy+4*scale,
                           cx+6*scale,cy-5*scale,fill=DARK_TEAL,
                           width=t+0.9,capstyle="round",joinstyle="round")
    elif name == "start":
        canvas.create_polygon(cx-5*scale,cy-9*scale,cx+9*scale,cy,
                              cx-5*scale,cy+9*scale,fill=color,outline="")
    elif name == "stop":
        rounded_polygon(canvas,(cx-7*scale,cy-7*scale,cx+7*scale,cy+7*scale),
                        2.5*scale,fill=color,outline="")
    elif name == "layers":
        for dy in (-5,1,7):
            canvas.create_line(cx-10*scale,cy+(dy-3)*scale,
                               cx,cy+(dy+2)*scale,cx+10*scale,cy+(dy-3)*scale,
                               fill=color,width=t,joinstyle="round")


class RoundedCard(tk.Frame):
    """Canvas rounded card with regular Tk controls inside."""
    def __init__(self, parent, *, pad=17, radius=22, color=WHITE, bg=BG):
        super().__init__(parent, bg=bg, borderwidth=0, highlightthickness=0)
        self.pad, self.radius, self.color = pad, radius, color
        self.canvas=tk.Canvas(self,bg=bg,bd=0,highlightthickness=0,width=12,height=12)
        self.canvas.pack(fill="both",expand=True)
        self.inner=tk.Frame(self.canvas,bg=color)
        self.window=self.canvas.create_window(pad,pad,window=self.inner,anchor="nw")
        self.canvas.bind("<Configure>",self._resize)

    def _resize(self,event):
        w,h=event.width,event.height
        self.canvas.delete("cardshape")
        if w<30 or h<30:
            return
        rounded_polygon(self.canvas,(4,6,w-3,h-2),self.radius,
                        fill="#CBDFEF",outline="",tags="cardshape")
        rounded_polygon(self.canvas,(1,1,w-4,h-6),self.radius,
                        fill=self.color,outline=BORDER,width=1,tags="cardshape")
        self.canvas.tag_lower("cardshape")
        self.canvas.coords(self.window,self.pad,self.pad)
        self.canvas.itemconfig(self.window,width=max(20,w-2*self.pad-4),
                               height=max(20,h-2*self.pad-7))


class PillButton(tk.Canvas):
    """Rounded button; callbacks are wired to the existing core actions."""
    def __init__(self,parent,title,command=None,*,fill=PALE_BLUE,fg=BLUE,
                 icon=None,height=43,font_size=11,bg=WHITE,enabled=True,interactive=True):
        super().__init__(parent,bg=bg,height=height,bd=0,highlightthickness=0,
                         cursor="hand2" if enabled and interactive else "arrow")
        self.title=title
        self.command=command
        self.fill,self.fg,self.icon=fill,fg,icon
        self.font_size=font_size
        self._enabled=enabled
        self.interactive=interactive
        self._hover=False
        self.bind("<Configure>",lambda _e:self.redraw())
        self.bind("<Button-1>",self._click)
        self.bind("<Enter>",self._enter)
        self.bind("<Leave>",self._leave)

    def _click(self,_event):
        if self._enabled and self.interactive and self.command:
            self.command()

    def _enter(self,_event):
        if self.interactive:
            self._hover=True
            self.redraw()

    def _leave(self,_event):
        if self.interactive:
            self._hover=False
            self.redraw()

    def set_label(self,value):
        self.title=value
        self.redraw()

    def set_enabled(self,value):
        self._enabled=bool(value)
        self.configure(cursor="hand2" if value and self.interactive else "arrow")
        self.redraw()

    def redraw(self):
        self.delete("all")
        w,h=self.winfo_width(),self.winfo_height()
        if w<18 or h<16:
            return
        fill=self.fill if self._enabled else "#EFF3F8"
        fg=self.fg if self._enabled else "#A3B2C3"
        radius=min(14,h//2)
        rounded_polygon(self,(2,5,w-2,h-1),radius,
                        fill="#D4E4EF" if self._enabled else "#EBF0F5",outline="")
        rounded_polygon(self,(2,2,w-2,h-4),radius,fill=fill,
                        outline=fg if self._hover and self._enabled else "",
                        width=1.3)
        if self.icon:
            # For the long confirmation caption, keep the caption centered.
            if self.title=="Konfirmasi Huruf / Enter":
                draw_icon(self,self.icon,35,h/2-1,fg,0.93)
                self.create_text(w/2+5,h/2-1,text=self.title,fill=fg,
                                 font=(FONT,self.font_size,"bold"))
            else:
                text_width=len(self.title)*self.font_size*.54
                start=max(22,(w-text_width-24)/2)
                draw_icon(self,self.icon,start,h/2-1,fg,0.83)
                self.create_text(start+18,h/2-1,text=self.title,fill=fg,
                                 font=(FONT,self.font_size,"bold"),anchor="w")
        else:
            self.create_text(w/2,h/2-1,text=self.title,fill=fg,
                             font=(FONT,self.font_size,"bold"))


class Toggle(tk.Canvas):
    def __init__(self,parent,on_change,*,bg=WHITE):
        super().__init__(parent,width=54,height=31,bg=bg,highlightthickness=0,cursor="hand2")
        self.enabled=False
        self.on_change=on_change
        self.bind("<Button-1>",self._click)
        self.render()

    def _click(self,_event):
        self.enabled=not self.enabled
        self.render()
        self.on_change(self.enabled)

    def render(self):
        self.delete("all")
        rounded_polygon(self,(2,3,52,28),13,
                        fill=TEAL if self.enabled else "#C4D1DF",outline="")
        x=38 if self.enabled else 16
        self.create_oval(x-10,5,x+10,25,fill=WHITE,outline="")


class StatusBadge(tk.Canvas):
    def __init__(self,parent,variable,*,width,bg,fg,icon):
        super().__init__(parent,width=width,height=51,bd=0,highlightthickness=0,
                         bg=BG)
        self.variable=variable
        self.bg_fill,self.fg,self.icon=bg,fg,icon
        self.variable.trace_add("write",lambda *_args:self.redraw())
        self.bind("<Configure>",lambda _e:self.redraw())

    def redraw(self):
        self.delete("all")
        w,h=self.winfo_width(),self.winfo_height()
        if w<15:
            return
        rounded_polygon(self,(3,6,w-3,h-1),16,fill="#D0E5F2",outline="")
        rounded_polygon(self,(2,2,w-3,h-5),16,fill=self.bg_fill,outline="")
        draw_icon(self,self.icon,29,h/2-1,self.fg,1.04)
        self.create_text(50,h/2-1,text=self.variable.get(),
                         font=(FONT,10,"bold"),fill=self.fg,anchor="w")


class MetricTile(tk.Canvas):
    def __init__(self,parent,title,variable,*,fill,fg,height=92):
        super().__init__(parent,bg=WHITE,height=height,bd=0,highlightthickness=0)
        self.title,self.variable=title,variable
        self.fill,self.fg=fill,fg
        self.variable.trace_add("write",lambda *_args:self.render())
        self.bind("<Configure>",lambda _e:self.render())

    def render(self):
        self.delete("all")
        w,h=self.winfo_width(),self.winfo_height()
        if w<20 or h<20:
            return
        rounded_polygon(self,(1,1,w-2,h-2),12,fill=self.fill,outline="")
        self.create_text(w/2,18,text=self.title,fill=NAVY,font=(FONT,10))
        self.create_text(w/2,max(38,(h+26)/2),text=self.variable.get(),
                         fill=self.fg,font=(FONT,21,"bold"))


class TextPanel(tk.Canvas):
    def __init__(self,parent,variable):
        super().__init__(parent,bg=WHITE,bd=0,highlightthickness=0,height=90)
        self.variable=variable
        self.variable.trace_add("write",lambda *_args:self.render())
        self.bind("<Configure>",lambda _e:self.render())

    def render(self):
        self.delete("all")
        w,h=self.winfo_width(),self.winfo_height()
        if w<24 or h<20:
            return
        rounded_polygon(self,(1,1,w-2,h-2),14,fill="#E3F0FF",outline="")
        self.create_text(w/2,h/2,text=self.variable.get(),fill=NAVY,
                         font=(FONT,20,"bold"),justify="center",width=max(70,w-28))


class BisindoApp:
    def __init__(self,args:argparse.Namespace):
        self.args=args
        self.root=tk.Tk()
        self.root.title("BISINDO — Pengenalan Alfabet dan Penyusunan Teks")
        sw=self.root.winfo_screenwidth()
        sh=self.root.winfo_screenheight()
        self.compact=sh<850
        width=min(1660,max(1040,sw-30))
        height=min(945,max(700,sh-10))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1040,700)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW",self.close)
        self.root.bind("<Return>",self._on_enter)
        self.root.bind("<Control-BackSpace>",self._on_backspace)
        self.root.bind("<Control-l>",self._on_clear)
        self.root.bind("<Control-L>",self._on_clear)

        # Webcam-only workflow. The inference worker and model are unchanged.
        self.composer=TextComposer(min_confirm_gap=0.7)
        self.worker=CameraWorker(args.model,args.landmarker,args.camera)
        self.auto_gate=AutoConfirmGate(hold_seconds=1.8)
        self.auto_enabled=False
        self._last_index=-1
        self._image=None
        self._placeholder_size=(0,0)
        self._stable_letter=None
        self._raw=tk.StringVar(value="—")
        self._stable=tk.StringVar(value="—")
        self._confidence=tk.StringVar(value="—")
        self._hands=tk.StringVar(value="0")
        self._text=tk.StringVar(value="(belum ada teks)")
        self._status=tk.StringVar(value="Kamera belum aktif")
        self._mode=tk.StringVar(value="MODE KAMERA")
        self._model_status=tk.StringVar(value="Model Belum Siap")
        self._cam_status=tk.StringVar(value="Kamera Nonaktif")

        placeholder=HERE/"assets"/"camera_placeholder.jpg"
        if not placeholder.exists():
            # Also supports testing this file directly outside its repo folder.
            placeholder=Path(__file__).resolve().parent/"assets"/"camera_placeholder.jpg"
        self._placeholder_source=Image.open(placeholder).convert("RGB")
        self._build()
        self._refresh_text()
        self._poll_job=self.root.after(90,self._poll)

    def _on_enter(self,_event):
        self.confirm()
        return "break"

    def _on_backspace(self,_event):
        self.backspace()
        return "break"

    def _on_clear(self,_event):
        self.clear()
        return "break"

    def _header(self):
        h=110 if self.compact else 135
        self.header=tk.Canvas(self.root,height=h,bg=BG,bd=0,highlightthickness=0)
        self.header.pack(fill="x",padx=0,pady=(0,6))
        self.header.pack_propagate(False)
        self.header.create_text(31,37 if self.compact else 44,
                                text="BISINDO",anchor="w",fill=NAVY,
                                font=(FONT,29 if self.compact else 33,"bold"))
        self.header.create_text(32,73 if self.compact else 82,
                                text="Pengenalan Alfabet dan Penyusunan Teks",
                                anchor="w",fill="#315786",
                                font=(FONT,12 if self.compact else 13))
        self.badge_frame=tk.Frame(self.header,bg=BG)
        self.badges=[]
        for var,w,bg,fg,ic in [
            (self._mode,206,"#FFF0C7","#A96100","camera"),
            (self._model_status,220,"#DAF8EC","#007C58","layers"),
            (self._cam_status,175,"#FCDDE5","#C12A43","video"),
        ]:
            badge=StatusBadge(self.badge_frame,var,width=w,bg=bg,fg=fg,icon=ic)
            badge.pack(side="left",padx=(0,9))
            self.badges.append(badge)
        self.badge_window=self.header.create_window(0,49 if self.compact else 58,
                                                     window=self.badge_frame,anchor="ne")
        self.header.bind("<Configure>",self._on_header_resize)

    def _on_header_resize(self,event):
        w,h=event.width,event.height
        if w<50 or h<20:
            return
        self.header.coords(self.badge_window,w-26,48 if self.compact else 58)

    def _build(self):
        self._header()
        outer=tk.Frame(self.root,bg=BG)
        outer.pack(fill="both",expand=True,padx=25,pady=(0,6))
        outer.grid_rowconfigure(0,weight=1)
        outer.grid_columnconfigure(0,weight=53,minsize=510,uniform="columns")
        outer.grid_columnconfigure(1,weight=47,minsize=480,uniform="columns")
        self.camera_card=RoundedCard(outer,pad=20 if not self.compact else 17)
        self.camera_card.grid(row=0,column=0,sticky="nsew",padx=(0,7))
        right=tk.Frame(outer,bg=BG)
        right.grid(row=0,column=1,sticky="nsew",padx=(7,0))
        right.grid_columnconfigure(0,weight=1)
        right.grid_rowconfigure(0,weight=11,minsize=215)
        right.grid_rowconfigure(1,weight=11,minsize=204)
        right.grid_rowconfigure(2,weight=7,minsize=125)
        self.pred_card=RoundedCard(right,pad=18)
        self.pred_card.grid(row=0,column=0,sticky="nsew",pady=(0,5))
        self.text_card=RoundedCard(right,pad=18)
        self.text_card.grid(row=1,column=0,sticky="nsew",pady=5)
        self.word_card=RoundedCard(right,pad=18)
        self.word_card.grid(row=2,column=0,sticky="nsew",pady=(5,0))
        self._build_camera()
        self._build_prediction()
        self._build_text()
        self._build_words()
        self._build_footer()

    def _section_title(self,parent,title,*,icon="camera",accent=BLUE,bar=False):
        # Keep the icon tile smaller than its parent row so the bottom edge
        # never gets clipped by Tk's fixed-height frame (notably on 768p).
        row_height=44 if self.compact else 50
        icon_size=40 if self.compact else 44
        row=tk.Frame(parent,bg=WHITE,height=row_height)
        row.pack(fill="x",pady=(0,5 if self.compact else 8))
        row.pack_propagate(False)
        if bar:
            tk.Frame(row,bg=TEAL,width=5).pack(side="left",fill="y",pady=5,padx=(0,9))
        holder=tk.Canvas(row,width=icon_size,height=icon_size,bg=WHITE,
                         bd=0,highlightthickness=0)
        holder.pack(side="left",padx=(0,12))
        rounded_polygon(holder,(1,1,icon_size-1,icon_size-1),10,
                        fill="#DBEDFF",outline="")
        draw_icon(holder,icon,icon_size/2,icon_size/2,accent,
                  1.0 if self.compact else 1.08)
        label(row,title,17 if self.compact else 19,NAVY,True).pack(side="left")
        return row

    def _build_camera(self):
        p=self.camera_card.inner
        header=self._section_title(p,"Tampilan Kamera",icon="camera",accent=BLUE,bar=True)
        start=PillButton(header,"Mulai",self.start_camera,fill=PALE_MINT,
                         fg=DARK_TEAL,icon="start",height=43,font_size=11)
        start.pack(side="right",padx=(0,5))
        start.configure(width=130)
        stop=PillButton(header,"Hentikan",self.stop_camera,fill=PALE_PINK,
                        fg="#C82742",icon="stop",height=43,font_size=11)
        stop.pack(side="right",padx=(0,8))
        stop.configure(width=115)

        self.preview=tk.Canvas(p,bd=0,highlightthickness=0,bg="#DEEEFF",height=220)
        self.preview.pack(fill="both",expand=True,padx=1,pady=(0,11))
        self.preview.bind("<Configure>",self._preview_resized)

        status_line=tk.Frame(p,bg=WHITE,height=32)
        status_line.pack(fill="x",pady=(0,10))
        status_line.pack_propagate(False)
        label(status_line,"● STATUS",10,"#00A48A",True).pack(side="left",padx=(5,15))
        tk.Label(status_line,textvariable=self._status,fg=MUTED,bg=WHITE,
                 font=(FONT,10),anchor="w").pack(side="left",fill="x",expand=True)

        bottom=tk.Frame(p,bg=WHITE,height=66 if self.compact else 74)
        bottom.pack(fill="x")
        bottom.pack_propagate(False)
        info=tk.Frame(bottom,bg="#EBF5FF")
        info.pack(side="left",fill="both",expand=True,padx=(0,12))
        infoline=tk.Frame(info,bg="#EBF5FF")
        infoline.pack(anchor="w",fill="x",padx=14,pady=(20 if self.compact else 23,0))
        label(infoline,"Konfirmasi huruf:",10,NAVY,True,bg="#EBF5FF").pack(side="left")
        label(infoline,"Enter",10,BLUE,True,bg="#D4E9FF",padx=9,pady=3).pack(side="left",padx=9)
        label(infoline,"(prediksi stabil)",9,MUTED,bg="#EBF5FF").pack(side="left")

        auto=tk.Frame(bottom,bg="#EBF5FF")
        auto.pack(side="right",fill="y")
        toggle_line=tk.Frame(auto,bg="#EBF5FF")
        toggle_line.pack(padx=12,pady=(11 if self.compact else 13,1))
        self.auto_toggle=Toggle(toggle_line,self._toggle_auto,bg="#EBF5FF")
        self.auto_toggle.pack(side="left",padx=(0,9))
        label(toggle_line,"Konfirmasi Otomatis",10,NAVY,True,bg="#EBF5FF").pack(side="left")
        label(auto,"Opsional • stabil 1,8 detik",9,MUTED,bg="#EBF5FF").pack(pady=(0,8))

    def _build_prediction(self):
        p=self.pred_card.inner
        self._section_title(p,"Prediksi Huruf",icon="bars")
        row=tk.Frame(p,bg=WHITE)
        row.pack(fill="both",expand=True,pady=(0,8))
        for i,(title,var,bg,fg) in enumerate([
            ("Langsung",self._raw,PALE_BLUE,BLUE),
            ("Stabil",self._stable,PALE_MINT,DARK_TEAL),
            ("Skor RF",self._confidence,PALE_GOLD,"#B66C00"),
            ("Tangan",self._hands,PALE_PINK,"#C52948"),
        ]):
            row.grid_columnconfigure(i,weight=1,uniform="metrics")
            tile=MetricTile(row,title,var,fill=bg,fg=fg,height=70 if self.compact else 92)
            tile.grid(row=0,column=i,sticky="nsew",padx=(0 if i==0 else 4,4 if i<3 else 0))
        row.grid_rowconfigure(0,weight=1)
        self.confirm_btn=PillButton(p,"Konfirmasi Huruf / Enter",self.confirm,
                                    icon="check",fill=TEAL,fg=WHITE,
                                    height=42 if self.compact else 55,
                                    font_size=12 if self.compact else 13)
        self.confirm_btn.pack(fill="x")

    def _build_text(self):
        p=self.text_card.inner
        self._section_title(p,"Teks Tersusun",icon="document")
        self.formed_text=TextPanel(p,self._text)
        if self.compact:
            self.formed_text.configure(height=66)
        # Anchor the action buttons above the card's bottom edge so the
        # rounded lower border and subtle button shadow are never clipped.
        # On short screens, the text display yields height to these controls.
        row=tk.Frame(p,bg=WHITE)
        row.pack(side="bottom",fill="x",pady=(0,6))
        for i,(name,fn,fill,fg,icon) in enumerate([
            ("Hapus",self.backspace,"#FFEACA","#AE6200","backspace"),
            ("Spasi",self.space,"#D6EFFF",BLUE,"space"),
            ("Bersihkan",self.clear,"#FBDDE5","#C52742","trash"),
        ]):
            row.grid_columnconfigure(i,weight=1,uniform="buttons")
            button=PillButton(row,name,fn,fill=fill,fg=fg,icon=icon,
                              height=39 if self.compact else 43,font_size=11)
            button.grid(row=0,column=i,sticky="ew",padx=(0 if i==0 else 5,5 if i<2 else 0))
        self.formed_text.pack(fill="both",expand=True,pady=(0,8))

    def _build_words(self):
        p=self.word_card.inner
        self._section_title(p,"Prediksi Kata Cerdas",icon="sparkles",accent=BLUE)
        row=tk.Frame(p,bg=WHITE)
        row.pack(fill="both",expand=True,pady=(0,4))
        self.word_buttons: list[PillButton] = []
        for i in range(4):
            row.grid_columnconfigure(i,weight=1,uniform="suggestions")
            button=PillButton(row,"—",None,fill=PALE_BLUE,fg=NAVY,
                              interactive=True,enabled=False,height=38 if self.compact else 48,
                              font_size=11)
            button.grid(row=0,column=i,sticky="nsew",padx=(0 if i==0 else 5,5 if i<3 else 0))
            self.word_buttons.append(button)
        row.grid_rowconfigure(0,weight=1)

    def _build_footer(self):
        self.footer=tk.Canvas(self.root,bg=BG,bd=0,highlightthickness=0,
                              height=51 if self.compact else 57)
        self.footer.pack(fill="x",padx=29,pady=(0,10))
        self.footer.bind("<Configure>",lambda _e:self._render_footer())
        self._status.trace_add("write",lambda *_args:self._render_footer())

    def _render_footer(self):
        self.footer.delete("all")
        w,h=self.footer.winfo_width(),self.footer.winfo_height()
        if w<24 or h<18:
            return
        rounded_polygon(self.footer,(0,1,w-1,h-2),14,fill="#DBFAF0",outline="#A8ECD8",width=1)
        self.footer.create_oval(23,h/2-12,47,h/2+12,fill="#0DB792",outline="")
        self.footer.create_line(29,h/2,34,h/2+5,42,h/2-5,
                                fill=WHITE,width=2.5,capstyle="round",joinstyle="round")
        self.footer.create_line(70,14,70,h-14,fill="#93DACB",width=2)
        self.footer.create_text(90,h/2,text=self._status.get(),anchor="w",
                                font=(FONT,10),fill=NAVY)

    def _toggle_auto(self,enabled):
        self.auto_enabled=enabled
        self.auto_gate.reset()
        self._status.set("Konfirmasi otomatis aktif: tahan huruf stabil selama 1,8 detik" if enabled
                         else "Konfirmasi otomatis nonaktif. Tekan Enter untuk mengonfirmasi.")

    def _preview_resized(self,event):
        if self._image is not None and self.worker.running():
            return
        w=max(20,event.width)
        h=max(20,event.height)
        if abs(w-self._placeholder_size[0])<4 and abs(h-self._placeholder_size[1])<4:
            return
        self._placeholder_size=(w,h)
        frame=ImageOps.fit(self._placeholder_source,(w,h),method=Image.Resampling.LANCZOS)
        self._show_preview(frame,placeholder=True)

    def _show_preview(self,img,*,placeholder=False):
        w=max(20,self.preview.winfo_width())
        h=max(20,self.preview.winfo_height())
        # Keep a single canvas so the preview itself can have rounded corners.
        bg=Image.new("RGB",(w,h),"#DCEEFF")
        if placeholder:
            img=ImageOps.fit(img,(w,h),method=Image.Resampling.LANCZOS)
            bg.paste(img,(0,0))
        else:
            # "Cover" the entire camera area, matching the placeholder.
            # The original frame remains untouched in CameraWorker for inference;
            # only the display may crop its outer edges to preserve proportions.
            resized=ImageOps.fit(img,(w,h),method=Image.Resampling.BILINEAR,
                                 centering=(0.5,0.5))
            bg.paste(resized,(0,0))
        mask=Image.new("L",(w,h),0)
        ImageDraw.Draw(mask).rounded_rectangle((0,0,w-1,h-1),radius=15,fill=255)
        rgba=bg.convert("RGBA")
        rgba.putalpha(mask)
        self._image=ImageTk.PhotoImage(rgba)
        self.preview.delete("all")
        self.preview.create_image(0,0,anchor="nw",image=self._image)
        if placeholder:
            self.preview.create_rectangle(1,h-37,w-1,h-1,fill="#D9EFFF",outline="")
            self.preview.create_text(w/2,h-19,text="Tampilan kamera akan muncul di sini",
                                     fill=NAVY,font=(FONT,12,"italic"))

    def start_camera(self):
        self.worker.start()
        self._cam_status.set("Menyalakan Kamera")
        self._model_status.set("Memuat Model")
        self._status.set("Memulai kamera dan model…")

    def stop_camera(self):
        self.worker.stop()
        self._stable_letter=None
        self.auto_gate.reset()
        self._raw.set("—")
        self._stable.set("—")
        self._confidence.set("—")
        self._hands.set("0")
        self._cam_status.set("Kamera Nonaktif")
        self._status.set("Kamera dihentikan")
        self._image=None
        self._placeholder_size=(0,0)
        self._preview_resized(type("Event",(),{"width":self.preview.winfo_width(),
                                               "height":self.preview.winfo_height()})())

    def _poll(self):
        shot=self.worker.snapshot()
        if shot.index!=self._last_index:
            self._last_index=shot.index
            self._stable_letter=shot.stable_letter
            self._raw.set(shot.raw_letter or "—")
            self._stable.set(shot.stable_letter or "—")
            self._confidence.set(f"{shot.score:.2f}" if shot.raw_letter else "—")
            self._hands.set(str(shot.hands))
            self._status.set(("Kesalahan: " + shot.status[7:] if shot.status.startswith("ERROR: ") else shot.status)
                             +(f"  |  {shot.fps:.1f} FPS" if shot.fps else ""))
            if shot.status.startswith("ERROR"):
                self._cam_status.set("Galat Kamera")
                self._model_status.set("Periksa Model")
                self.auto_gate.reset()
            elif shot.status.startswith("Kamera aktif") or shot.rgb is not None:
                self._cam_status.set("Kamera Aktif")
                self._model_status.set("Model Siap")
            if shot.rgb is not None:
                frame=Image.fromarray(shot.rgb)
                self._show_preview(frame)
            if self.auto_enabled and self.auto_gate.update(shot.stable_letter):
                if self.composer.confirm(shot.stable_letter):
                    self._refresh_text()
                    self._status.set(f"Konfirmasi otomatis: huruf {shot.stable_letter} ditambahkan")
            elif not self.auto_enabled:
                self.auto_gate.reset()
        self._poll_job=self.root.after(90,self._poll)

    def _refresh_text(self):
        self._text.set(self.composer.text.upper() if self.composer.text else "(belum ada teks)")
        self._update_suggestions()

    def _update_suggestions(self):
        if not hasattr(self, "word_buttons"):
            return
        suggestions = suggest_words(self.composer.text, limit=4)
        for i in range(4):
            btn = self.word_buttons[i]
            if i < len(suggestions):
                word = suggestions[i]
                btn.set_label(word.upper())
                btn.command = lambda w=word: self._on_select_suggestion(w)
                btn.set_enabled(True)
            else:
                btn.set_label("—")
                btn.command = None
                btn.set_enabled(False)

    def _on_select_suggestion(self, word: str):
        self.composer.replace_last_word(word)
        self._status.set(f"Kata '{word.upper()}' dipilih")
        self._refresh_text()

    def confirm(self):
        if not self._stable_letter:
            self._status.set("Belum ada prediksi stabil. Tahan gestur hingga stabil.")
            return
        if self.composer.confirm(self._stable_letter):
            self.auto_gate.mark_manual_confirm(self._stable_letter)
            self._status.set(f"Huruf {self._stable_letter} ditambahkan")
            self._refresh_text()
        else:
            self._status.set("Konfirmasi terlalu cepat. Coba lagi sesaat.")

    def backspace(self):
        self.composer.backspace()
        self._refresh_text()

    def space(self):
        self.composer.add_space()
        self._refresh_text()

    def clear(self):
        self.composer.clear()
        self._refresh_text()

    def close(self):
        if getattr(self,"_poll_job",None) is not None:
            try:
                self.root.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job=None
        self.worker.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser=argparse.ArgumentParser(description="BISINDO: pengenalan alfabet dan penyusunan teks")
    parser.add_argument("--model",default="models/random_forest.joblib")
    parser.add_argument("--landmarker",default="models/hand_landmarker.task")
    parser.add_argument("--camera",type=int,default=0)
    BisindoApp(parser.parse_args()).run()


if __name__=="__main__":
    main()
