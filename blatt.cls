\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{blatt}[2013/03/29 LaTeX class]

\DeclareOption{twoside}{
  \PassOptionsToClass{\CurrentOption}{scrartcl}
}
\DeclareOption*{
  \ClassWarning{blatt}{Unbekannte Option '\CurrentOption'}
}

\ProcessOptions\relax

\LoadClass[a4paper,ngerman]{scrartcl}

\RequirePackage{ifxetex}
\ifxetex\else\RequirePackage[utf8]{inputenc}\fi
\RequirePackage[ngerman]{babel}
\RequirePackage{amsmath,amsthm,amssymb,amscd,color,graphicx}
\RequirePackage{environ}
\RequirePackage{framed}
\RequirePackage[protrusion=true,expansion=true]{microtype}
\RequirePackage{lmodern}
\RequirePackage{multicol}
\RequirePackage{hyperref}

\addto\captionsngerman{\renewcommand\tablename{Tafel}}

\setlength{\unitlength}{1cm}

%\setlength\parskip{\medskipamount}
%\setlength\parindent{0pt}

\newlength{\titleskip}
\setlength{\titleskip}{1.3em}
\newcommand{\makeheader}{%
  \thispagestyle{empty}%
  \begin{picture}(0,0)
   \put(0,-1){\vbox{%
     \noindent Mathematik der Vorhersagen \hfill
     JGW-Schülerakademie Papenburg-1 2015}}
   \put(13,-2){}  %{\includegraphics[scale=0.10]{../cover}}
  \end{picture}%
  \vspace*{1.7cm}}
\renewcommand{\maketitle}[1]{%
  \makeheader
  \begin{center}\Large \textbf{\textsf{#1}}\end{center}
  \vspace{\titleskip}}

\renewcommand*\theenumi{\alph{enumi}}
\renewcommand{\labelenumi}{\theenumi)}

\newlength{\aufgabenskip}
\setlength{\aufgabenskip}{1.4em}
\newcounter{aufgabennummer}
\newenvironment{aufgabeUnshaded}[1]{
  \refstepcounter{aufgabennummer}
  \textbf{Aufgabe \theaufgabennummer.} \emph{#1} \par
}{\vspace{\aufgabenskip}}
\makeatletter
\NewEnviron{aufgabe}[1]{%
  \refstepcounter{aufgabennummer}%
  \begin{shaded}%
    \@startsection{paragraph}{4}{\z@}%
      {-0.00001ex}%
      {\medskipamount}%
      {}%
      {\textbf{Aufgabe \theaufgabennummer.} \emph{#1}}%
    %\noindent\subparagraph{\textbf{Aufgabe \theaufgabennummer.} \emph{#1}}
    \BODY%
  \end{shaded}%
}
\makeatother
\newcommand{\fixlistspacing}{\vspace{-\medskipamount}\vspace{-\itemsep}}

\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000

\definecolor{hicolor}{rgb}{.55,.12,.55}
\newcommand{\hil}[1]{\textcolor{hicolor}{#1}}

\definecolor{shadecolor}{rgb}{.93,.93,.93}
