"""One-shot helper: writes paper/fig_stack.tex with the self-play learning loop.
Run: python scripts/_write_fig_stack.py
"""
from pathlib import Path

TEX = r"""% Standalone preview of the architecture figure for visual debugging.
% Build:  pdflatex fig_stack.tex
% Render: pdftoppm -png -r 200 fig_stack.pdf fig_stack
\documentclass[border=10pt,tikz]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes, arrows.meta, positioning, fit, backgrounds, calc}
\begin{document}
\begin{tikzpicture}[
    >={Stealth[length=2.2mm]},
    font=\footnotesize,
    box/.style ={draw, rounded corners=2pt, align=center,
                 minimum height=0.85cm, inner sep=4pt},
    data/.style={box, fill=gray!10,    minimum width=2.6cm, font=\scriptsize},
    sym/.style ={box, fill=blue!10,    minimum width=3.6cm},
    learn/.style={box, fill=orange!18, minimum width=3.6cm, font=\scriptsize},
    agent/.style={box, fill=green!12,  minimum width=2.6cm, font=\scriptsize},
      grp/.style ={draw=black!35, dashed, rounded corners=4pt, inner sep=10pt},
    flow/.style={->, semithick, blue!55!black},
    twoway/.style={<->, semithick, gray!70!black},
    backflow/.style={->, semithick, dashed, red!65!black},
    spflow/.style={->, semithick, orange!75!black},
    spflowdash/.style={->, semithick, dashed, orange!65!black},
    kgflow/.style={->, semithick, dashed, violet!65!black},
    elabel/.style={font=\scriptsize\itshape, fill=white,
                   inner xsep=2pt, inner ysep=1pt}
  ]

  % ---- Column 1: data sources -----------------------------------------
  \node[data] (scry)  at (0,  3.6) {Scryfall\\(card DB)};
  \node[data] (sb)    at (0,  1.2) {Commander\\Spellbook};
  \node[data] (rules) at (0, -1.2) {Comprehensive\\Rules};
  \node[data] (decks) at (0, -3.6) {Decklists\\(\texttt{data/decks})};

  % ---- Column 2: symbolic substrate ------------------------------------
  \node[sym] (kg)  at (4.7,  2.4) {\textbf{Knowledge Graph}\\
        Neo4j + n10s (OWL2)\\+ APOC + GraphSAGE};
  \node[sym] (eng) at (4.7,  0.0) {\textbf{Game Engine}\\
        zones, stack, priority,\\phases, combat, triggers,\\state-based actions};
  \node[sym] (rb)  at (4.7, -2.6) {\textbf{Replay Buffer}\\
        trajectories $(s_t,a_t,r_t)$};

  % ---- Column 3: world model V/M/C ------------------------------------
  \node[learn] (V) at (9.7,  2.4) {V: state encoder\\(Set-Transformer + KG ctx)};
  \node[learn] (M) at (9.7,  0.0) {M: MDN-LSTM dynamics\\+ JEPA target head};
  \node[learn] (C) at (9.7, -2.4) {C: controller\\(policy + value heads)};
  \begin{pgfonlayer}{background}
    \node[grp, fit=(V)(M)(C),
          label={[font=\small\bfseries, yshift=-1pt]above:World Model}] (wm) {};
  \end{pgfonlayer}

  % ---- Self-play node (below WM, lighter orange tint) -----------------
  \node[learn, fill=orange!8, minimum width=3.6cm, font=\scriptsize]
        (sp) at (9.7, -4.6) {Self-play\\(agents $\times$ engine)};

  % ---- Column 4: agents ------------------------------------------------
  \node[agent] (a1) at (14.0,  3.0) {Random / Heuristic};
  \node[agent] (a2) at (14.0,  1.8) {KG-Heuristic};
  \node[agent] (a3) at (14.0,  0.6) {Ollama LLM};
  \node[agent] (a4) at (14.0, -0.6) {World-Model};
  \node[agent] (a5) at (14.0, -1.8) {Active Inference};
  \node[agent] (a6) at (14.0, -3.0) {LLM-Fusion};
  \begin{pgfonlayer}{background}
    \node[grp, fit=(a1)(a6),
          label={[font=\small\bfseries, yshift=-1pt]above:Agent Zoo}] (agents) {};
  \end{pgfonlayer}

  % ---- data -> symbolic -----------------------------------------------
  \draw[flow] (scry.east)  -- (2.3,  3.6) -- (2.3,  3.0) -- ([yshift= 0.6cm]kg.west)
              node[elabel, pos=0.83, above=1pt]{cards};
  \draw[flow] (sb.east)    -- (2.3,  1.2) -- (2.3,  1.8) -- ([yshift=-0.6cm]kg.west)
              node[elabel, pos=0.83, above=1pt]{combos};
  \draw[flow] (rules.east) -- (2.3, -1.2) -- (2.3,  0.6) -- ([yshift= 0.6cm]eng.west)
              node[elabel, pos=0.82, above=1pt]{rules};
  \draw[flow] (decks.east) -- (2.3, -3.6) -- (2.3, -0.6) -- ([yshift=-0.6cm]eng.west)
              node[elabel, pos=0.80, above=1pt]{game cfg};

  % ---- engine <-> KG (card lookup) ------------------------------------
  \draw[twoway] (kg.south) -- node[elabel, xshift=-2pt]{card lookup} (eng.north);

  % ---- engine -> replay buffer ----------------------------------------
  \draw[flow] (eng.south) -- node[elabel]{$(s_t,a_t,r_t)$} (rb.north);

  % ---- world-model training: replay -> M/V; KG ctx -> V ---------------
  \draw[flow] (rb.east) -- ++(0.4,0) |-
      node[elabel,pos=0.74, above=1pt]{train (JEPA)} (M.west);
  \draw[flow] (rb.east) -- ++(0.4,0) |-
      node[elabel,pos=0.26, below=1pt]{features} (V.west);
  \draw[flow] (kg.east) -- node[elabel, above=1pt]{KG ctx} (V.west);

  % ---- self-play loop -------------------------------------------------
  % Controller policy drives self-play episodes
  \draw[spflowdash]
      (C.south) -- (9.7,-3.55) -- (sp.north)
      node[elabel, pos=0.5, right=2pt]{policy};

  % Agent zoo contributes decisions during self-play
  \draw[spflowdash]
      ([xshift=0.4cm]a6.south) -- (12.5,-4.6) -- (sp.east)
      node[elabel, pos=0.5, above=1pt]{decisions};

  % Self-play generates trajectories -> replay buffer (solid orange)
  \draw[spflow]
      (sp.west) -- (5.5,-4.6)
      node[elabel, above=1pt, pos=0.55]{trajectories $(s_t,a_t,r_t)$}
      |- (rb.south);

  % Self-play learned evidence -> KG extension layer (dashed violet)
  \draw[kgflow]
      (sp.west) -- (3.6,-4.6)
      |- ([yshift=-0.25cm]kg.south)
      node[elabel, pos=0.68, right=2pt]{learned\\evidence};

  % ---- agent decide\_action loop --------------------------------------
  % Top bus (y=4.6) for state/legal_actions; bottom bus (y=-6.0) for
  % action return, placed below the self-play node.
  \path (eng.north east) ++(0.7,0) coordinate (engNEgap);
  \path (eng.south east) ++(0.7,0) coordinate (engSEgap);

  % engine -> agent (state, legal_actions)
  \draw[flow]
      (eng.north east) -- (engNEgap)
      -- (7.2, 4.6) -- (14.0, 4.6)
        node[elabel, pos=0.55, above=2pt]{state, legal\_actions}
      -- (a1.north);

  % agent -> engine (action): bottom bus at y=-6.0 (below sp node)
  \draw[backflow]
      ([xshift=-0.4cm]a6.south) -- (13.6,-6.0)
      -- (7.2,-6.0)
        node[elabel, pos=0.55, below=2pt]{action}
      -- (engSEgap)
      -- (eng.south east);

  % Agent-side lookups (agent->KG, agent->C) omitted for clarity;
  % described in the body text.

\end{tikzpicture}
\end{document}
"""

out = Path(__file__).parent.parent / "paper" / "fig_stack.tex"
out.write_text(TEX, encoding="utf-8")
print(f"Written {out} ({TEX.count(chr(10))} lines)")
