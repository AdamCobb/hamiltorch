%%
%% This is file `jmlr.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% jmlr.dtx  (with options: `jmlr.cls,package')
%% 
%%  jmlr.dtx
%%  Copyright 2022 Nicola Talbot
%% 
%%  This work may be distributed and/or modified under the
%%  conditions of the LaTeX Project Public License, either version 1.3
%%  of this license or (at your option) any later version.
%%  The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%%  and version 1.3 or later is part of all distributions of LaTeX
%%  version 2005/12/01 or later.
%% 
%%  This work has the LPPL maintenance status `maintained'.
%% 
%%  The Current Maintainer of this work is Nicola Talbot.
%% 
%%  This work consists of the files jmlr.dtx and jmlr.ins and the derived files jmlrutils.sty, jmlr.cls, jmlrbook.cls.
%% 
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%% First editor:
%% Second editor:
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{jmlr}[2022/02/09 v1.30 (NLCT) Journal of Machine Learning Research]
\RequirePackage{xkeyval}
\RequirePackage{calc}
\RequirePackage{etoolbox}
\providecommand*{\jmlrprehyperref}{}
\newif\if@openright
\newif\if@mainmatter \@mainmattertrue
\@ifundefined{ifgrayscale}{
  \newif\ifgrayscale
  \grayscalefalse
}{}
\DeclareOptionX{color}{\grayscalefalse
  \PassOptionsToPackage{color}{xcolor}}
\DeclareOptionX{gray}{\grayscaletrue
  \PassOptionsToPackage{gray}{xcolor}}
\DeclareOptionX{draft}{\PassOptionsToClass{\CurrentOption}{article}}
\DeclareOptionX{final}{\PassOptionsToClass{\CurrentOption}{article}}
\newif\iftablecaptiontop
\tablecaptiontoptrue
\newcommand{\tableconts}[3]{%
  \iftablecaptiontop
    #2\label{#1}\vskip\baselineskip
    {\centering #3\par}%
  \else
    {\centering #3\par}%
    \vskip\baselineskip
    #2\label{#1}%
  \fi
}
\DeclareOptionX{tablecaptiontop}{\tablecaptiontoptrue}
\DeclareOptionX{tablecaptionbottom}{\tablecaptiontopfalse}
\define@choicekey{jmlr.cls}{tablecaption}[\val\nr]{top,bottom}{%
  \ifcase\nr\relax
    \tablecaptiontoptrue
  \or
    \tablecaptiontopfalse
  \fi
}
\newif\ifjmlrhtml
\jmlrhtmlfalse
\DeclareOptionX{html}{%
 \ClassWarning{jmlr}{html option is now deprecated}%
 \jmlrhtmltrue}
\DeclareOptionX{nohtml}{\jmlrhtmlfalse}
\def\pt@size{11pt}
\DeclareOptionX{10pt}{\renewcommand{\pt@size}{10pt}}
\DeclareOptionX{11pt}{\renewcommand{\pt@size}{11pt}}
\DeclareOptionX{12pt}{\renewcommand{\pt@size}{12pt}}
\newcommand*{\@jmlrproceedings}{Journal of Machine Learning Research}
\newcommand*{\@jmlrabbrvproceedings}{JMLR}
\newcommand*{\jmlrproceedings}[2]{%
  \renewcommand*{\@jmlrabbrvproceedings}{#1}%
  \renewcommand*{\@jmlrproceedings}{#2}%
}
\newcommand*{\jmlrnowcp}{%
  \jmlrproceedings{JMLR}{Journal of Machine Learning Research}%
}
\newcommand*{\jmlrwcp}{%
  \jmlrproceedings{JMLR W\&CP}{JMLR: Workshop and Conference Proceedings}%
}
\newcommand*{\jmlrpmlr}{%
  \jmlrproceedings{PMLR}{Proceedings of Machine Learning Research}%
}
 \DeclareOptionX{mlmain}{\jmlrproceedings{PMLR}{Under Review - Proceedings Track}\newcommand*{\test}{Proceedings Track}}
 \DeclareOptionX{mlabstract}{\jmlrproceedings{PMLR}{Under Review - Extended Abstract Track}\newcommand*{\test}{Extended Abstract Track}}
\DeclareOptionX{nowcp}{\jmlrnowcp}
\DeclareOptionX{wcp}{\jmlrwcp}
\DeclareOptionX{pmlr}{\jmlrpmlr}
\DeclareOptionX{cleveref}{\PassOptionsToPackage{cleveref}{jmlrutils}}
\DeclareOptionX{oneside}{\@twosidefalse \@mparswitchfalse}
\DeclareOptionX{twoside}{\@twosidetrue \@mparswitchtrue}
\@twosidetrue
\newif\ifviiXx
\viiXxfalse
\DeclareOptionX{7x10}{\viiXxtrue}
\DeclareOptionX{letterpaper}{\PassOptionsToPackage{letterpaper}{typearea}}
\DeclareOptionX*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{letterpaper}
\ProcessOptionsX
\if@twoside
 \PassOptionsToClass{twoside}{article}
\fi
\LoadClass[\pt@size]{article}
\ifviiXx
  \setlength{\paperwidth}{7in}
  \setlength{\paperheight}{10in}
  \setlength{\textwidth}{5.25in}
  \setlength{\textheight}{8.2in}
  \setlength{\topmargin}{0.4in}
  \setlength{\headheight}{0.2in}
  \setlength{\headsep}{0.2in}
  \setlength{\hoffset}{-1in}
  \setlength{\voffset}{-1in}
  \setlength{\evensidemargin}{0.75in}
  \setlength{\oddsidemargin}{1.0in}
\else
  \setlength{\oddsidemargin}{0.25in}
  \setlength{\evensidemargin}{0.25in}
  \setlength{\marginparwidth}{0.07 true in}
  \setlength{\topmargin}{-0.5in}
  \addtolength{\headsep}{0.25in}
  \setlength{\textheight}{8.5 true in}
  \setlength{\textwidth}{6.0 true in}
\fi
\AtEndDocument{\@jmlrenddoc}
\RequirePackage{placeins}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{natbib}
\RequirePackage{graphicx}
\RequirePackage{url}
\PassOptionsToPackage{x11names}{xcolor}
\RequirePackage{xcolor}
\PassOptionsToPackage{algo2e,ruled}{algorithm2e}
\RequirePackage{algorithm2e}
\setlength\algomargin{0pt}
\RequirePackage{jmlrutils}
\jmlrprehyperref
\@ifundefined{@pre@hyperref}{}{\@pre@hyperref}
\RequirePackage{hyperref}
\RequirePackage{nameref}
\@ifundefined{@post@hyperref}{}{\@post@hyperref}
\hypersetup{colorlinks,
            linkcolor=blue,
            citecolor=blue,
            urlcolor=magenta,
            linktocpage,
            plainpages=false}
\ifgrayscale
  \hypersetup{draft}
\fi
\renewcommand{\topfraction}{0.95} % let figure take up nearly whole page
\renewcommand{\textfraction}{0.05} % let figure take up nearly whole page
\widowpenalty=10000\relax
\clubpenalty=10000\relax
\@mparswitchtrue
\bibliographystyle{plainnat}
\bibpunct{(}{)}{;}{a}{,}{,}
\renewcommand{\section}{\@startsection{section}{1}{\z@}%
   {-0.24in \@plus -1ex \@minus -.2ex}%
   {0.10in \@plus.2ex}%
   {\normalfont\rmfamily\bfseries\large\raggedright}%
}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
   {-0.20in \@plus -1ex \@minus -.2ex}%
   {0.08in \@plus .2ex}%
   {\normalfont\rmfamily\bfseries\normalsize\raggedright}%
}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
   {-0.18in \@plus -1ex \@minus -.2ex}%
   {0.08in \@plus .2ex}%
   {\normalfont\normalsize\rmfamily\mdseries\scshape\raggedright}%
}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
   {1.5ex plus 0.5ex minus .2ex}%
   {-1em}%
   {\normalfont\normalsize\rmfamily\bfseries}%
}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\z@}%
   {1.5ex plus 0.5ex minus .2ex}%
   {-1em}%
   {\normalfont\normalsize\rmfamily\bfseries\itshape}}
\renewcommand*\@seccntformat[1]{%
  \csname pre#1num\endcsname
  \csname the#1\endcsname.\enskip
}
\renewcommand*{\@makefntext}[1]{%
  \@setpar
  {%
    \@@par
    \@tempdima\hsize
    \advance \@tempdima -15pt\relax
    \parshape \@ne 15pt \@tempdima
  }%
  \par
  \parindent 2em\noindent
  \hbox to \z@ {\hss {\@thefnmark }\footnoteseptext\hfil }#1%
}
\newcommand*{\footnoteseptext}{. }
\renewcommand*{\thanks}[1]{%
  \refstepcounter{mpfootnote}%
  \footnotemark[\number\value{mpfootnote}]%
  \xappto\@thanks{\noexpand\footnotetext[\number\value{mpfootnote}]{#1}}%
}
\ifjmlrhtml
  \renewenvironment{abstract}{\HCode{<h3>}Abstract\HCode{</h3>}}{}%
\else
  \renewenvironment{abstract}
  {{\centering\large\bfseries Abstract\par}\vspace{0.7ex}%
    \bgroup
       \leftskip 20pt\rightskip 20pt\small\noindent\ignorespaces}%
  {\par\egroup\vskip 0.25ex}
\fi
\newenvironment{keywords}
{\bgroup\leftskip 20pt\rightskip 20pt \small\noindent{\bfseries
Keywords:} \ignorespaces}%
{\par\egroup\vskip 0.25ex}
\newlength\aftertitskip     \newlength\beforetitskip
\newlength\interauthorskip  \newlength\aftermaketitskip
%% Changeable parameters.
\setlength\aftertitskip{0.1in plus 0.2in minus 0.2in}
\setlength\beforetitskip{0.05in plus 0.08in minus 0.08in}
\setlength\interauthorskip{0.08in plus 0.1in minus 0.1in}
\setlength\aftermaketitskip{0.3in plus 0.1in minus 0.1in}
\newcommand*{\titlebreak}{\newline}
\newcommand*{\titletag}[1]{}
\renewcommand*{\title}[2][\@title]{%
  \def\@shorttitle{#1}%
  \def\@title{#2}%
  \protected@write\@auxout{}{\string\jmlr@title{#1}{#2}}%
  \jmlrtitlehook
}
\newcommand*{\@shorttitle}{\jobname}
\newcommand*{\jmlrtitlehook}{}
\newcommand*{\jmlr@title}[2]{}
\renewcommand*{\author}[2][]{%
  \def\@author{#2}%
  \def\@sauthor{#1}%
  \def\@jmlr@aux@author{#2}\@onelevel@sanitize\@jmlr@aux@author
  \ifx\@sauthor\@empty
    \let\@jmlr@aux@sauthor\@jmlr@aux@author
  \else
    \let\@shortauthor\@sauthor
    \def\@jmlr@aux@sauthor{#1}\@onelevel@sanitize\@jmlr@aux@sauthor
  \fi
  \jmlrauthorhook
  \protected@write\@auxout
    {}{\string\jmlr@author{\@jmlr@aux@sauthor}{\@jmlr@aux@author}}%
}
\newcommand*{\jmlrauthorhook}{}
\newcommand*{\jmlr@author}[2]{}
\newcommand*{\@shortauthor}{}
\newcommand*{\@firstauthor}{}
\newcommand*{\@firstsurname}{}
\newlength\jmlrlength
\def\jmlrmaketitle{%
 \jmlrpremaketitlehook
 \def\@jmlr@authors@sep{, }%
 \par
 \begingroup
   \def\footnoteseptext{ }%
   \def\thempfn{\textsuperscript{\thefootnote}}%
   \def\thefootnote{\fnsymbol{footnote}}%
   \if@twocolumn
     \twocolumn[\@jmlrmaketitle]%
   \else
     \@jmlrmaketitle
   \fi
   \@thanks
 \endgroup
\label{jmlrstart}%
\ifx\@sauthor\@empty
  \settowidth{\jmlrlength}{\@evenhead}%
  \ifdim\jmlrlength>\textwidth
    \def\@shortauthor{\@firstsurname\space et al.}%
  \fi
\fi
\settowidth{\jmlrlength}{\@titlefoot}%
\ifdim\jmlrlength>\textwidth
  \def\@jmlrauthors{\@firstauthor\space \emph{et al}}%
\fi
\jmlrmaketitlehook
\thispagestyle{jmlrtps}%
\setcounter{footnote}{0}%
\let\maketitle\relax \let\@maketitle\relax
\gdef\@thanks{}\gdef\@author{}\let\thanks\@gobble
\def\@jmlr@authors@sep{ \& }%
}
\newcommand*{\jmlrmaketitlehook}{}
\newcommand*{\jmlrpremaketitlehook}{}
\newcommand{\jmlrhtmlmaketitle}{%
  \ifx\@jmlr@authors\@empty
    \sbox\jmlrbox{\let\addr\relax\@author}%
  \fi
  \noindent\HCode{<h2>}\@title\HCode{</h2>}
  \noindent\@jmlr@authors
}
%%\begin{macro}{\jmlrbox}
\newsavebox\jmlrbox
\ifjmlrhtml
  \let\maketitle\jmlrhtmlmaketitle
\else
  \let\maketitle\jmlrmaketitle
\fi
\def\@startauthor{\noindent \normalsize\bfseries}
\def\@endauthor{}
\def\@starteditor{\noindent \small {\bfseries \@edname:~}}
\def\@endeditor{\normalsize}
\def\jmlrpretitle{\vskip\beforetitskip\begin{center}\Large\bfseries}
\def\jmlrposttitle{\par\end{center}\vskip\aftertitskip}
\newcommand*{\nametag}[1]{}
\def\jmlrpreauthor{%
\bgroup
  \def\nametag##1{##1}%
  \def\and{\unskip\enspace{\normalfont and}\enspace}%
  \def\addr{\mdseries\small\itshape}%
  \def\name{\ClassError{jmlr}{Use \string\Name{Author's Name} not \string\name}{}}%
  \def\email{\ClassError{jmlr}{Use \string\Email{address} not \string\email}{}}%
  \def\AND{\@endauthor\normalfont\hss \vskip \interauthorskip
    \@startauthor}%
  \@startauthor
}
\newcommand{\addr}{}
\def\@email{\hfill\small\mdseries\scshape}%
\def\@name{\normalsize\upshape\bfseries}%
\def\@parsename#1 #2\end@parsename{%
  \def\@tmp{#2}%
  \ifx\@tmp\@nnil
    \def\@surname{#1}%
    \let\@nextparsename\@parsenamenoop
  \else
    \@getinitial#1-\relax\relax\end@getinitial
    \ifx\@forenames\@empty
      \def\@forenames{#1}%
      \protected@edef\@initials{\@initial}%
    \else
      \expandafter\toks@\expandafter{\@forenames}%
      \edef\@forenames{\space\the\toks@}%
      \expandafter\toks@\expandafter{\@initials}%
      \protected@edef\@initials{\the\toks@\@initial}%
    \fi
    \let\@nextparsename\@parsename
  \fi
  \@nextparsename#2\end@parsename
}
\def\@parsenamenoop#1\end@parsename{}
\def\@getinitial#1#2-#3#4\end@getinitial{%
  \def\@jmlr@tmp{#3}%
  \if\@jmlr@tmp\relax
    \def\@initial{#1.}%
  \else
    \def\@initial{#1.-#3.}%
  \fi
}
\newcommand*{\Name}[2][]{%
  \def\@authorlist{#1}%
  \def\@forenames{}%
  \def\@surname{}%
  \def\nametag##1{}%
  \@parsename#2 \@nil\end@parsename
  \ifx\@shortauthor\@empty
    \ifx\@sauthor\@empty
      \global\let\@shortauthor\@surname
      \global\let\@firstsurname\@surname
    \fi
    \ifx\@authorlist\@empty
      \protected@xdef\@jmlrauthors{\@initials\space\@surname}%
    \else
      \protected@xdef\@jmlrauthors{\@authorlist}%
    \fi
    \global\let\@firstauthor\@jmlrauthors
  \else
    \ifx\@sauthor\@empty
      \expandafter\toks@\expandafter{\@shortauthor}%
      \protected@xdef\@shortauthor{\the\toks@\space\@surname}%
    \fi
    \ifx\@authorlist\@empty
      \ifx\@jmlrauthors\@empty
        \protected@xdef\@jmlrauthors{\@initials\space\@surname}%
      \else
        \protected@xdef\@jmlrauthors{\@jmlrauthors
          \noexpand\@jmlr@authors@sep
          \@initials\space\@surname}%
      \fi
    \else
      \ifx\@jmlrauthors\@empty
        \protected@xdef\@jmlrauthors{\@authorlist}%
      \else
        \protected@xdef\@jmlrauthors{\@jmlrauthors
          \noexpand\@jmlr@authors@sep
          \@authorlist
        }%
      \fi
    \fi
  \fi
  \def\nametag##1{##1}%
  \@name #2%
}
\newcommand*{\jmlrabbrnamelist}[1]{%
  \def\nametag##1{}%
  \def\@jmlr@authors@sep{, }%
  \def\@jmlr@namelist{}%
  \@for\@thisname:=#1\do{%
    \expandafter\@jmlrabbrname\expandafter{\@thisname}%
    \ifx\@jmlr@namelist\@empty
       \protected@edef\@jmlr@namelist{%
          \@initials\space\@surname
       }%
    \else
       \protected@edef\@jmlr@namelist{%
          \@jmlr@namelist
          \noexpand\@jmlr@authors@sep
          \@initials\space\@surname
       }%
    \fi
  }%
  \def\@jmlr@authors@sep{ \& }%
  \@jmlr@namelist
}
\newcommand*{\@jmlrabbrname}[1]{%
  \def\@initials{}%
  \def\@surname{}%
  \def\@forenames{}%
  \@parsename#1 \@nil\end@parsename
}
\newcommand*{\Email}[1]{{\@email #1}}
\def\jmlrpostauthor{\@endauthor\egroup
  \par
  \vskip \aftermaketitskip
  \noindent
  \ifx\@editor\@empty
  \else
    \@starteditor \@editor \@endeditor
  \fi
  \vskip \aftermaketitskip
}
\def\@jmlrmaketitle{%
 {%
  \jmlrpretitle
  {%
    \def\titletag##1{##1}%
    \@title
  }%
  \jmlrposttitle
  \jmlrpreauthor \ignorespaces\@author \jmlrpostauthor
  }%
}
\newcommand*\kernelmachines{(for
  {\textsc{http://www.kernel-machines.org}})}
\newcommand*{\editorname}{Editor}
\newcommand*{\editorsname}{Editors}
\let\@edname\editorname
\def\@editor{}
\def\editor#1{%
  \global\let\@edname\editorname
  \gdef\@editor{#1}%
}
\def\editors#1{%
  \global\let\@edname\editorsname
  \gdef\@editor{#1}%
}
\def\firstpageno#1{\setcounter{page}{#1}}
\@ifundefined{startpage}{}{\firstpageno{\startpage}}
\newcommand*{\@jmlrenddoc}{%
  \FloatBarrier
  \phantomsection
  \protected@edef\@currentlabelname{end of \@shorttitle}%
  \label{jmlrend}\null
  \global\let\@reprint\@empty
}
\newcommand*{\@titlefoot}{\scriptsize\copyright\space\@jmlryear
    \space\@jmlr@authors.\space\hfill
    \@reprint
}
\let\@reprint\@empty
\newcommand{\reprint}[1]{%
  \gdef\@reprint{Reprinted with permission for JMLR#1}}
\newcommand\ps@jmlrtps{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{\scriptsize \@jmlrproceedings
    \ifx\@jmlrvolume\@empty
    \else
       \space\@jmlrvolume
       \ifx\@jmlrissue\@empty\else(\@jmlrissue)\fi
       \ifx\@jmlrpages\@empty
          \ifx\@jmlryear\@empty
          \else
             \if\@jmlrissue\@empty,\fi
          \fi
       \else
          :%
       \fi
    \fi
    \ifx\@jmlrpages\@empty
    \else
       \ifx\@jmlrvolume\@empty\space\fi
       \@jmlrpages
       \ifx\@jmlryear\@empty\else,\fi
    \fi
    \ifx\@jmlryear\@empty\else\space\@jmlryear\fi
    \hfill
    \ifx\@jmlrworkshop\@empty
      \ifx\@jmlrsubmitted\@empty
      \else
        Submitted \@jmlrsubmitted
        \ifx\@jmlrpublished\@empty\else;\fi
      \fi
      \ifx\@jmlrpublished\@empty
      \else
        \space Published \@jmlrpublished
      \fi
    \else
      \space\@jmlrworkshop
    \fi
  }%
  \let\@evenhead\@oddhead
  \def\@oddfoot{\@titlefoot}%
  \let\@evenfoot\@oddfoot
}
\def\ps@jmlrps{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{\hfill {\small\scshape \@shorttitle} \hfill}%
  \def\@oddfoot{\hfill \small\rmfamily \thepage \hfill}%
  \def\@evenhead{\hfill {\small\scshape \@shortauthor} \hfill}%
  \def\@evenfoot{\hfill \small\rmfamily \thepage \hfill}%
}%
\pagestyle{jmlrps}
\providecommand*{\@jmlrvolume}{}
\newcommand*{\jmlrvolume}[1]{\renewcommand*{\@jmlrvolume}{#1}}
\providecommand*{\@jmlrissue}{}
\newcommand*{\jmlrissue}[1]{\renewcommand*{\@jmlrissue}{#1}}
\providecommand*{\@jmlryear}{}
\newcommand*{\jmlryear}[1]{\renewcommand*{\@jmlryear}{#1}}
\providecommand*\@jmlrpages{\pageref{jmlrstart}--\pageref{jmlrend}}
\newcommand*{\jmlrpages}[1]{\renewcommand*{\@jmlrpages}{#1}}
\providecommand*\@jmlrsubmitted{}
\newcommand*{\jmlrsubmitted}[1]{\renewcommand*{\@jmlrsubmitted}{#1}}
\providecommand*\@jmlrpublished{}
\newcommand*{\jmlrpublished}[1]{\renewcommand*{\@jmlrpublished}{#1}}
\providecommand*\@jmlrworkshop{}
\newcommand*{\jmlrworkshop}[1]{%
 \renewcommand*{\@jmlrworkshop}{#1}%
 \protected@write\@auxout{}{\string\jmlr@workshop{#1}}%
}
\newcommand*{\jmlr@workshop}[1]{}
\renewcommand*{\date}[1]{%
 \renewcommand*{\@date}{#1}%
 \protected@write\@auxout{}{\string\jmlr@date{#1}}%
}
\newcommand*{\jmlr@date}[1]{}
\newcommand*{\@jmlrauthors}{}
\newcommand*{\@jmlr@authors}{\@jmlrauthors}
\newcommand*{\jmlrauthors}[1]{\global\def\@jmlr@authors{#1}}

\def\figurecaption#1#2{\noindent\hangindent 40pt
                       \hbox to 36pt {\small\slshape #1 \hfil}
                       \ignorespaces {\small #2}}
\def\figurecenter#1#2{\centerline{{\slshape #1} #2}}
\def\figurecenter#1#2{\centerline{{\small\slshape #1} {\small #2}}}
\long\def\@makecaption#1#2{%
   \vskip 10pt
   \setbox\@tempboxa\hbox{#1: #2}%
   \ifdim \wd\@tempboxa >\hsize               % IF longer than one line:
       \begin{list}{#1:}{%
       \settowidth{\labelwidth}{#1:}
       \setlength{\leftmargin}{\labelwidth}
       \addtolength{\leftmargin}{\labelsep}
        }\item #2 \end{list}\par   % Output in quote mode
     \else                                    %   ELSE  center.
       \hbox to\hsize{\hfil\box\@tempboxa\hfil}
   \fi}
\def\abovestrut#1{\rule[0in]{0in}{#1}\ignorespaces}
\def\belowstrut#1{\rule[-#1]{0in}{#1}\ignorespaces}
\newcommand{\acks}[1]{\section*{Acknowledgments}#1}
\newcommand{\researchnote}[1]{\noindent {\LARGE\itshape Research Note} #1}
\providecommand{\ifprint}[2]{\ifgrayscale#1\else#2\fi}
\ifjmlrhtml
\else
  \let\@org@Ginclude@graphics\Ginclude@graphics
  \def\@jmlr@old@Ginclude@graphics#1{%
    \begingroup
    \let\input@path\Ginput@path
    \ifprint{\filename@parse{#1-gray}}{\filename@parse{#1}}%
    \ifx\filename@ext\relax
      \@for\Gin@temp:=\Gin@extensions\do{%
        \ifx\Gin@ext\relax
          \Gin@getbase\Gin@temp
        \fi}%
    \else
      \ifprint{\filename@parse{#1}}{}%
      \Gin@getbase{\Gin@sepdefault\filename@ext}%
      \ifx\Gin@ext\relax
         \@warning{File `#1' not found}%
         \def\Gin@base{\filename@area\filename@base}%
         \edef\Gin@ext{\Gin@sepdefault\filename@ext}%
      \fi
    \fi
      \ifx\Gin@ext\relax
        \ifprint{\@org@Ginclude@graphics{#1}}%
        {%
           \@latex@error{File `#1' not found}%
           {I could not locate the file with any of these extensions:^^J%
            \Gin@extensions^^J\@ehc}%
        }%
      \else
         \@ifundefined{Gin@rule@\Gin@ext}%
           {\ifx\Gin@rule@*\@undefined
              \@latex@error{Unknown graphics extension: \Gin@ext}\@ehc
            \else
              \expandafter\Gin@setfile\Gin@rule@*{\Gin@base\Gin@ext}%
             \fi}%
           {\expandafter\expandafter\expandafter\Gin@setfile
               \csname Gin@rule@\Gin@ext\endcsname{\Gin@base\Gin@ext}}%
      \fi
    \endgroup}
\def\@jmlr@new@Ginclude@graphics#1{%
  \ifx\detokenize\@undefined\else
    \edef\Gin@extensions{\detokenize\expandafter{\Gin@extensions}}%
  \fi
  \begingroup
  \let\input@path\Ginput@path
  \ifprint{\set@curr@file{#1-gray}}{\set@curr@file{#1}}%
  \expandafter\filename@parse\expandafter{\@curr@file}%
  \ifx\filename@ext\Gin@gzext
    \expandafter\filename@parse\expandafter{\filename@base}%
    \ifx\filename@ext\relax
      \let\filename@ext\Gin@gzext
    \else
      \edef\Gin@ext{\Gin@ext\Gin@sepdefault\Gin@gzext}%
    \fi
  \fi
  \let\@jmlr@filename@ext\filename@ext
  \ifx\filename@ext\relax
    \@for\Gin@temp:=\Gin@extensions\do{%
      \ifx\Gin@ext\relax
        \Gin@getbase\Gin@temp
      \fi}%
    \ifprint
    {\ifx\Gin@ext\relax
       \set@curr@file{#1}%
       \expandafter\filename@parse\expandafter{\@curr@file}%
       \ifx\filename@ext\Gin@gzext
        \expandafter\filename@parse\expandafter{\filename@base}%
        \ifx\filename@ext\relax
          \let\filename@ext\Gin@gzext
        \else
          \edef\Gin@ext{\Gin@ext\Gin@sepdefault\Gin@gzext}%
        \fi
       \fi
       \let\@jmlr@filename@ext\filename@ext
       \ifx\filename@ext\relax
       \@for\Gin@temp:=\Gin@extensions\do{%
         \ifx\Gin@ext\relax
           \Gin@getbase\Gin@temp
         \fi}%
       \fi
     \fi}{}%
  \fi
  \ifx\@jmlr@filename@ext\relax
  \else
    \Gin@getbase{\Gin@sepdefault\filename@ext}%
    \ifx\Gin@ext\relax
      \let\Gin@savedbase\filename@base
      \let\Gin@savedext\filename@ext
      \edef\filename@base{\filename@base\Gin@sepdefault\filename@ext}%
      \let\filename@ext\relax
       \@for\Gin@temp:=\Gin@extensions\do{%
          \ifx\Gin@ext\relax
            \Gin@getbase\Gin@temp
          \fi}%
        \ifx\Gin@ext\relax
        \let\filename@base\Gin@savedbase
        \let\filename@ext\Gin@savedext
      \fi
    \fi
    \ifx\Gin@ext\relax
       \@warning{File `#1' not found}%
       \def\Gin@base{\filename@area\filename@base}%
       \edef\Gin@ext{\Gin@sepdefault\filename@ext}%
    \fi
  \fi
  \ifx\Gin@ext\relax
       \@latex@error{File `#1' not found}%
       {I could not locate the file with any of these extensions:^^J%
        \Gin@extensions^^J\@ehc}%
  \else
     \@ifundefined{Gin@rule@\Gin@ext}%
       {\ifx\Gin@rule@*\@undefined
          \@latex@error{Unknown graphics extension: \Gin@ext}\@ehc
        \else
          \expandafter\Gin@setfile\Gin@rule@*{\Gin@base\Gin@ext}%
         \fi}%
       {\expandafter\expandafter\expandafter\Gin@setfile
           \csname Gin@rule@\Gin@ext\endcsname{\Gin@base\Gin@ext}}%
  \fi
  \endgroup
}
  \@ifpackagelater{graphics}{2019/07/01}
  {\let\Ginclude@graphics\@jmlr@new@Ginclude@graphics}%
  {\let\Ginclude@graphics\@jmlr@old@Ginclude@graphics}%
\fi
\newcommand{\artappendix}{\par
  \setcounter{section}{0}
  \setcounter{subsection}{0}
  \def\thesection{\Alph{section}}
  \def\theHsection{\theHchapter.\Alph{section}}
  \def\presectionnum{Appendix~}%
}
\let\appendix\artappendix
\newcommand{\booklinebreak}[1][]{}
\newcounter{chapter}
\renewcommand\thechapter{\@arabic\c@chapter}
\newcommand\@chapapp{\chaptername}
\@addtoreset{section}{chapter}
\newcommand*\chaptermark[1]{}
\newcommand\bookchapter{%
  \if@openright\cleardoublepage\else\clearpage\fi
                    \thispagestyle{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\newcommand\artchapter{%
  \ClassError{jmlr}{Chapters not permitted in articles}{}}
\let\chapter\artchapter
\def\@chaptoclabel{chapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \refstepcounter{chapter}%
                       \if@mainmatter
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{\@chaptoclabel}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{\@chaptoclabel}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{\@chaptoclabel}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\newcommand{\chaptertitleformat}[1]{%
  \Huge\bfseries#1%
}
\newcommand{\chapternumberformat}[1]{%
  \huge\bfseries \@chapapp\space#1\par\nobreak
  \vskip 20\p@
}
\newcommand*{\chapterformat}{\raggedright}
\newlength\postchapterskip
\setlength\postchapterskip{40pt}
\newlength\prechapterskip
\setlength\prechapterskip{50pt}
\def\@makechapterhead#1{%
  \null\vskip\prechapterskip
  {\parindent \z@ \normalfont\chapterformat
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \chapternumberformat{\thechapter}%
      \fi
    \fi
    \interlinepenalty\@M
    \chaptertitleformat{#1}\par\nobreak
    \vskip \postchapterskip
  }}
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
\def\@makeschapterhead#1{%
  \vspace*{\prechapterskip}%
  {\parindent \z@
    \normalfont\chapterformat
    \interlinepenalty\@M
    \chaptertitleformat{#1}\par\nobreak
    \vskip \postchapterskip
  }}
\newcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \large\bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\let\l@appendix\l@chapter
\newcommand\chaptername{Chapter}
\newcommand\frontmatter{%
  \cleardoublepage
  \@mainmatterfalse
  \renewcommand*{\theHchapter}{front-\thechapter}%
  \pagenumbering{roman}%
  \morefrontmatter
}
\newcommand\morefrontmatter{}
\newcommand\mainmatter{%
  \cleardoublepage
  \@mainmattertrue
  \setcounter{chapter}{0}%
  \renewcommand*{\theHchapter}{\thechapter}%
  \pagenumbering{arabic}%
  \moremainmatter
}
\newcommand\moremainmatter{}
\newcommand\backmatter{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \@mainmatterfalse}
\newcommand*{\booktocpreamble}{}
\newcommand*{\booktocpostamble}{}
\newcommand\booktableofcontents{%
  \if@twocolumn
    \@restonecoltrue\onecolumn
  \else
    \@restonecolfalse
  \fi
  \chapter*{\contentsname
    \@mkboth{\MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
  \booktocpreamble
  \@starttoc{toc}%
  \booktocpostamble
  \if@restonecol
    \twocolumn
  \else
    \clearpage
  \fi
  \@mkboth{}{}%
}
\let\arttableofcontents\tableofcontents
\newcommand{\artpart}{%
  \def\toclevel@part{0}%
  \if@noskipsec \leavevmode\fi
  \par
  \addvspace{4ex}%
  \@afterindentfalse
  \secdef\@artpart\@sartpart
}
\let\@artpart\@part
\let\@sartpart\@spart
\newcommand\bookpart{%
  \def\toclevel@part{-1}%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \thispagestyle{plain}%
  \if@twocolumn
    \onecolumn
    \@tempswatrue
  \else
    \@tempswafalse
  \fi
  \preparthook
  \secdef\@bookpart\@sbookpart}
\newcommand{\parttitleformat}[1]{%
  \Huge\bfseries#1%
}
\newcommand*{\@parttoclabel}{part}
\def\@partapp{\partname}
\newcommand{\partnumberformat}[1]{%
  \Huge\bfseries \@partapp\nobreakspace#1\par\nobreak
  \vskip 20\p@
}
\newcommand{\preparthook}{\null\vfil}
\newcommand*{\partformat}{\centering}
\def\@bookpart[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{\@parttoclabel}{\protect\numberline{\thepart}#1}%
    \else
      \addcontentsline{toc}{\@parttoclabel}{#1}%
    \fi
    \markboth{}{}%
    {\interlinepenalty \@M
     \normalfont\partformat
     \ifnum \c@secnumdepth >-2\relax
       \partnumberformat{\thepart}%
     \fi
     \parttitleformat{#2}\par}%
    \postparthook}
\def\@sbookpart#1{%
    {\interlinepenalty \@M
     \normalfont\partformat
     \parttitleformat{#1}\par}%
    \postparthook}
\def\postparthook{\vfil\newpage
              \if@twoside
               \if@openright
                \null
                \thispagestyle{empty}%
                \newpage
               \fi
              \fi
              \if@tempswa
                \twocolumn
              \fi}
\newcommand\bookappendix{\par
  \setcounter{table}{0}%
  \setcounter{figure}{0}%
  \zeroextracounters
  \par
  \gdef\theHchapter{\Alph {chapter}}%
  \xdef\Hy@chapapp{\Hy@appendixstring}%
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}%
  \def\@write@jmlr@import{\@@write@jmlr@apdimport}%
  \csname appendixmore\endcsname
}
\newcommand*{\jmlrbookcommands}{%
  \let\part\bookpart
  \let\chapter\bookchapter
  \let\appendix\bookappendix
  \let\tableofcontents\booktableofcontents
  \def\thesection{\thechapter.\arabic{section}}%
}
\newcommand*{\jmlrarticlecommands}{%
  \let\part\artpart
  \let\chapter\artchapter
  \let\appendix\artappendix
  \let\tableofcontents\arttableofcontents
  \def\thesection{\arabic{section}}%
}
\newcommand*{\@jmlr@check@warn}[1]{}
\newcommand*{\@jmlr@check@packages}{%
  \@ifpackageloaded{epsfig}{%
    \ClassError{jmlr}{Obsolete package `epsfig' detected.
     \MessageBreak
     Please use \string\includegraphics\space to include images
     instead}{}}{}%
  \@ifpackageloaded{psfig}{%
    \ClassError{jmlr}{Obsolete package `psfig' detected.
     \MessageBreak
     Please use \string\includegraphics\space to include images
     instead}{}}{}%
  \@ifpackageloaded{subfig}{%
    \@jmlr@check@warn{Package `subfig' detected.\MessageBreak
    This will cause a conflict if the article is incorporated
    \MessageBreak
    into a book using jmlbook.cls.
    \MessageBreak
    Please use \string\subfigure\space and
    \string\subtable\space instead}}{}%
  \@ifpackageloaded{theorem}{%
   \ClassError{jmlr}{Package `theorem' detected.\MessageBreak
    This can cause a conflict with other packages used by jmlr}{}}{}%
  \@ifpackageloaded{ntheorem}{%
   \ClassError{jmlr}{Package `ntheorem' detected.\MessageBreak
    This can cause a conflict with other packages used by jmlr}{}}{}%
  \@ifpackageloaded{amsthm}{%
   \ClassError{jmlr}{Package `amsthm' detected.\MessageBreak
    This package conflicts with the jmlr class}{}}{}%
  \@ifpackageloaded{pdfpages}%
   {\@jmlr@check@warn{Package `pdfpages' detected.\MessageBreak
   This can cause a problem for jmlrbook}}{}%
  \@ifpackageloaded{geometry}%
   {\@jmlr@check@warn{Package `geometry' detected.\MessageBreak
   This can cause a problem for jmlrbook}}{}%
  \@ifpackageloaded{tabularx}{%
   \ClassError{jmlr}{Package `tabularx' detected.\MessageBreak
    This will break footnote links}{}}{}%
  \@ifpackageloaded{jmlr2e}{%
   \ClassError{jmlr}{Package `jmlr2e' detected.\MessageBreak
    This can't be used with the jmlr class}{}}{}%
}
\AtBeginDocument{%
\@jmlr@check@packages
\let\@jmlr@check@packages\relax
}
\newcommand*{\jmlrSuppressPackageChecks}{%
  \let\@jmlr@check@packages\relax
}
\DeclareRobustCommand*{\obsoletefontcs}[1]{%
  \ClassWarning{jmlr}{Obsolete command
    \expandafter\string\csname#1\endcsname\space detected}%
  \csname #1 \endcsname
}
\renewcommand*{\bf}{%
  \obsoletefontcs{bf}%
}
\renewcommand*{\it}{%
  \obsoletefontcs{it}%
}
\renewcommand*{\sc}{%
  \obsoletefontcs{sc}%
}
\renewcommand*{\rm}{%
  \obsoletefontcs{rm}%
}
\renewcommand*{\sf}{%
  \obsoletefontcs{sf}%
}
\renewcommand*{\tt}{%
  \obsoletefontcs{tt}%
}
\providecommand*{\jmlrcheckforpseudocode}{%
  \@ifpackageloaded{pseudocode}%
  {%
    \let\pseudoRETURN\RETURN
    \let\pseudoTRUE\TRUE
    \let\pseudoFALSE\FALSE
    \let\pseudoAND\AND
    \let\pseudoOR\OR
    \let\pseudoNOT\NOT
    \let\pseudoTO\TO
    \let\pseudoCOMMENT\COMMENT
    \let\pseudoIF\IF
    \let\pseudoELSE\ELSE
    \let\pseudoFOR\FOR
    \let\pseudoFORALL\FORALL
    \let\pseudoWHILE\WHILE
    \let\pseudoREPEAT\REPEAT
    \let\pseudoUNTIL\UNTIL
    \let\pseudoENDFOR\ENDFOR
    \let\RETURN\undefined
    \let\TRUE\undefined
    \let\FALSE\undefined
    \let\AND\undefined
    \let\OR\undefined
    \let\NOT\undefined
    \let\TO\undefined
    \let\COMMENT\undefined
    \let\IF\undefined
    \let\ELSE\undefined
    \let\FOR\undefined
    \let\FORALL\undefined
    \let\WHILE\undefined
    \let\REPEAT\undefined
    \let\UNTIL\undefined
    \let\ENDFOR\undefined
    \preto\pseudocode{%
    \let\RETURN\pseudoRETURN
    \let\TRUE\pseudoTRUE
    \let\FALSE\pseudoFALSE
    \let\AND\pseudoAND
    \let\OR\pseudoOR
    \let\NOT\pseudoNOT
    \let\TO\pseudoTO
    \let\COMMENT\pseudoCOMMENT
    \let\IF\pseudoIF
    \let\ELSE\pseudoELSE
    \let\FOR\pseudoFOR
    \let\FORALL\pseudoFORALL
    \let\WHILE\pseudoWHILE
    \let\REPEAT\pseudoREPEAT
    \let\UNTIL\pseudoUNTIL
    \let\ENDFOR\pseudoENDFOR
    }%
  }%
  {}%
}
\jmlrcheckforpseudocode
\endinput
%%
%% End of file `jmlr.cls'.
