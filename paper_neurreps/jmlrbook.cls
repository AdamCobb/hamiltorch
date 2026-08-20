%%
%% This is file `jmlrbook.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% jmlr.dtx  (with options: `jmlrbook.cls,package')
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
\ProvidesClass{jmlrbook}[2022/02/09 v1.30 (NLCT) JMLR Book Style]
\RequirePackage{xkeyval}
\RequirePackage{setspace}
\newcommand*\jmlrprefacepath{}
\IfFileExists{currfile.sty}%
{
  \RequirePackage{currfile}
  \renewcommand*\jmlrprefacepath{\currfilepath}
}%
{%
  \RequirePackage{fink}
  \ifdef\finkpath
  {%
    \renewcommand*\jmlrprefacepath{\finkpath}%
  }
  {%
    \ClassWarning{jmlrbook}{`currfile' package required}
  }
}
\providecommand*{\jmlrprehyperref}{}
\newif\ifgrayscale
\grayscalefalse
\DeclareOptionX{draft}{\setlength\overfullrule{5pt}}
\DeclareOptionX{final}{\setlength\overfullrule{0pt}}
\DeclareOptionX{color}{\grayscalefalse}
\DeclareOptionX{gray}{\grayscaletrue}
\DeclareOptionX{letterpaper}{\PassOptionsToClass{\CurrentOption}{jmlr}}
\DeclareOptionX{7x10}{\PassOptionsToClass{\CurrentOption}{jmlr}}
\DeclareOptionX{html}{\PassOptionsToClass{\CurrentOption}{jmlr}}
\DeclareOptionX{nohtml}{\PassOptionsToClass{\CurrentOption}{jmlr}}
\newcommand*{\jmlrprefaceheader}{%
  \phantomsection
  \chapter*{\prefacename}%
  \addcontentsline{toc}{chapter}{\prefacename}%
  \markboth{\prefacename}{\prefacename}%
}
\DeclareOptionX{wcp}{%
  \PassOptionsToClass{\CurrentOption}{jmlr}%
}
\DeclareOptionX{pmlr}{%
  \PassOptionsToClass{\CurrentOption}{jmlr}%
}
\DeclareOptionX{nowcp}{%
  \PassOptionsToClass{\CurrentOption}{jmlr}%
}
\DeclareOptionX{tablecaptiontop}{\PassOptionsToClass{\CurrentOption}{jmlr}}
\DeclareOptionX{tablecaptionbottom}{\PassOptionsToClass{\CurrentOption}{jmlr}}
\DeclareOptionX{10pt}{\PassOptionsToClass{\CurrentOption}{jmlr}}
\DeclareOptionX{11pt}{\PassOptionsToClass{\CurrentOption}{jmlr}}
\DeclareOptionX{12pt}{\PassOptionsToClass{\CurrentOption}{jmlr}}
\@twosidetrue
\DeclareOptionX{oneside}{\PassOptionsToClass{\CurrentOption}{jmlr}}
\DeclareOptionX{twoside}{\PassOptionsToClass{\CurrentOption}{jmlr}}
\define@boolkey{jmlrbook.cls}[jmlr]{pdfxa}[true]{}
\jmlrpdfxafalse
\ProcessOptionsX
\@ifundefined{jmlrgrayscale}{}%
{%
  \ifnum\jmlrgrayscale=0\relax
    \grayscalefalse
  \else
    \grayscaletrue
  \fi
}
\ifgrayscale
  \newcommand*{\jmlrwritepdfinfo}{%
     \protected@write\@auxout{}{\string\jmlrbook@info{\xmpAuthor}{\xmpTitle}}%
  }
  \ifjmlrpdfxa
   \def\convertDate{\getYear}
   {\catcode`\D=12
    \gdef\getYear D:#1#2#3#4{\edef\xYear{#1#2#3#4}\getMonth}
   }
   \def\getMonth#1#2{\edef\xMonth{#1#2}\getDay}
   \def\getDay#1#2{\edef\xDay{#1#2}\getHour}
   \def\getHour#1#2{\edef\xHour{#1#2}\getMin}
   \def\getMin#1#2{\edef\xMin{#1#2}\getSec}
   \def\getSec#1#2{\edef\xSec{#1#2}\getTZh}
   {%
     \catcode`\Z=12
     \gdef\tmpz{Z}
   }
  \def\hash{\expandafter\@gobble\string\#}%
  \def\amp{\expandafter\@gobble\string\&}%
  \def\xmpAmp{\amp\hash x0026;}%
  \def\sep{</rdf:li><rdf:li>}
  \def\TextCopyright{\amp\hash x00A9;}
  \def\Title#1{\gdef\xmpTitle{#1}}
  \def\Author#1{\gdef\xmpAuthor{#1}}
  \def\Keywords#1{\gdef\xmpKeywords{#1}}
   \let\xmpKeywords\@empty
   \let\xmpSubject\xmpKeywords
  \def\Creator#1{\gdef\xmpCreator{#1}}
   \def\xmpCreator{\@pdfcreator}
  \def\Producer#1{\gdef\xmpProducer{#1}}
   \def\xmpProducer{pdfTeX}
  \def\Volume#1{\gdef\xmpVolume{#1}}
   \let\xmpVolume\@empty
  \def\Issue#1{\gdef\xmpIssue{#1}}
   \let\xmpIssue\@empty
  \def\CoverDisplayDate#1{\gdef\xmpCoverDisplayDate{#1}}
   \let\xmpCoverDisplayDate\@empty
  \def\CoverDate#1{\gdef\xmpCoverDate{#1}}
   \let\xmpCoverDate\@empty
  \def\Copyright#1{\gdef\xmpCopyright{#1}}
   \let\xmpCopyright\@empty
  \def\Doi#1{\gdef\xmpDoi{#1}}
   \let\xmpDoi\@empty
  \def\Lastpage#1{\gdef\xmpLastpage{#1}}
   \let\xmpLastpage\@empty
  \def\Firstpage#1{\gdef\xmpFirstpage{#1}}
   \let\xmpFirstpage\@empty
  \def\Journaltitle#1{\gdef\xmpJournaltitle{#1}}
   \let\xmpJournaltitle\@empty
  \def\Journalnumber#1{\gdef\xmpJournalnumber{#1}}
   \let\xmpJournalnumber\@empty
  \def\Org#1{\gdef\xmpOrg{#1}}
   \let\xmpOrg\@empty
  \def\CreatorTool#1{\gdef\xmpCreatorTool{#1}}
   \def\xmpCreatorTool{\xmpProducer}
  \def\AuthoritativeDomain#1{\gdef\xmpAuthoritativeDomain{#1}}
   \let\xmpAuthoritativeDomain\@empty
  \def\findUUID#1{\edef\tmpstring{\pdfmdfivesum{#1}}
       \expandafter\eightofnine\tmpstring\end}
  \def\eightofnine#1#2#3#4#5#6#7#8#9\end{%
       \xdef\eightchars{#1#2#3#4#5#6#7#8}
       \fouroffive#9\end}
  \def\fouroffive#1#2#3#4#5\end{\xdef\ffourchars{#1#2#3#4}
       \sfouroffive#5\end}
  \def\sfouroffive#1#2#3#4#5\end{\xdef\sfourchars{#1#2#3#4}
       \tfouroffive#5\end}
  \def\tfouroffive#1#2#3#4#5\end{\xdef\tfourchars{#1#2#3#4}
       \xdef\laststring{#5}}
  \def\uuid{\eightchars-%
            \ffourchars-%
            \sfourchars-%
            \tfourchars-%
            \laststring}
  \def\getTZh#1{%
    \def\TZprefix{#1}%
    \ifx\TZprefix\tmpz
      \def\xTZsign{+}%
      \def\xTZh{00}%
      \def\xTZm{00}%
      \let\getTZnext\doConvDate
    \else
      \let\xTZsign\TZprefix
      \let\getTZnext\getTZhm
    \fi
    \getTZnext
  }
  \def\getTZhm#1#2'#3#4'{%
      \edef\xTZh{#1#2}%
      \edef\xTZm{#3#4}%
    \doConvDate
  }
  \def\doConvDate{%
      \edef\convDate{\xYear-\xMonth-\xDay
        T\xHour:\xMin:\xSec\xTZsign\xTZh:\xTZm}%
  }
  \newcommand{\@pre@hyperref}{%
   \IfFileExists{FOGRA39L.icc}%
   {%
     \pdfminorversion=3
     \pdfpageattr{/MediaBox[0 0 595 793]
                  /BleedBox[0 0 595 793]
                  /TrimBox[25 20 570 773]}%
      \findUUID{\jobname.pdf}%
      \edef\xmpdocid{\uuid}%
      \findUUID{\pdfcreationdate}%
      \edef\xmpinstid{\uuid}%
      \InputIfFileExists{\jobname.xmpdata}{}{}%
      \RequirePackage{xmpincl}%
      \expandafter\convertDate\pdfcreationdate
      \def\@pctchar{\expandafter\@gobble\string\%}
      \def\@bchar{\expandafter\@gobble\string\\}
      \immediate\pdfobj stream attr{/N 4}  file{FOGRA39L.icc}
      \edef\OBJ@CVR{\the\pdflastobj}
      \pdfcatalog{/OutputIntents [ <<
       /Type/OutputIntent
       /S/GTS_PDFX
       /OutputCondition (FOGRA39)
       /OutputConditionIdentifier (FOGRA39 \@bchar(ISO Coated v2
        300\@pctchar\space \@bchar(ECI\@bchar)\@bchar))
       /DestOutputProfile \OBJ@CVR\space 0 R
       /RegistryName(http://www.color.org)
      >> ]}
      \input glyphtounicode.tex
      \input glyphtounicode-cmr.tex
      \pdfgentounicode=1
      \RequirePackage[draft,pdftex,pdfpagemode=UseNone,bookmarks=false]{hyperref}%
    }%
    {%
       \ClassError{jmlrbook}{Can't find `FOGRA39L.icc'}%
         {Download ISOcoated\string_v2\string_330\string_bas.icc from
         http://www.colormanagement.org/en/isoprofile.html
         Rename it FOGRA39L.icc and put it in the pdfx folder}%
    }%
  }
  \renewcommand*{\jmlrwritepdfinfo}{%
      \begingroup
        \let\&=\xmpAmp
        \IfFileExists{pdfx-1a.xmp}{%
           \pdfcompresslevel=0
         \immediate\pdfobj stream attr {/Type /Metadata /Subtype /XML}
         file{pdfx-1a.xmpi}
         \pdfcatalog{/Metadata \the\pdflastobj\space 0 R}
        }%
        {}%
      \endgroup
      \protected@write\@auxout{}{\string\jmlrbook@info{\xmpAuthor}{\xmpTitle}}%
      \pdfinfo{
          /Author(\xmpAuthor)%
          /Title(\xmpTitle)%
          /Creator(\xmpProducer)%
          /CreationDate(\convDate)%
          /ModDate(\convDate)%
          /Producer(\xmpProducer)%
          /Trapped /False
          /GTS_PDFXVersion (PDF/X-1:2001)%
          /GTS_PDFXConformance (PDF/X-1a:2001)%
      }%
  }
  \fi
\else
  \newcommand*{\jmlrwritepdfinfo}{}
\fi
\newcommand*{\jmlrbook@info}[2]{}
\newcommand*{\jmlrbook@location}[1]{}
\newcommand*{\@post@hyperref}{%
  \let\@org@c@lenddoca\c@lenddoca
  \let\c@lenddoca\undefined
}
\let\@org@LoadClass\LoadClass
\def\LoadClass#1{\let\LoadClass\@org@LoadClass\@org@LoadClass{jmlr}}
\@org@LoadClass{combine}
\let\c@lenddoca\@org@c@lenddoca
\RequirePackage{combnat}
\renewcommand\c@laNAT@parse[1]{{%
     \let\protect=\@unexpandable@protect\let~\relax
     \let\active@prefix=\@gobble
     \xdef\NAT@temp{\csname b@#1\@extra@b@citeb\endcsname}}%
     \expandafter\NAT@split\NAT@temp?????@@%
     \expandafter\NAT@parse@date\NAT@date??????@@%
     \ifciteindex\NAT@index\fi}

\renewcommand\c@lbNAT@parse[1]{{%
     \let\protect=\@unexpandable@protect\let~\relax
     \let\active@prefix=\@gobble
     \xdef\NAT@temp{\csname B?\jobname?@#1\@extra@b@citeb\endcsname}}%
     \expandafter\NAT@split\NAT@temp?????@@%
     \expandafter\NAT@parse@date\NAT@date??????@@%
     \ifciteindex\NAT@index\fi}
\newif\if@openright
\@openrighttrue
\newif\if@mainmatter
\newcommand*{\pagerule}[1][0pt]{\par\noindent
  \rule[#1]{\linewidth}{2pt}\par}
\ifjmlrhtml
  \newenvironment{preface}[1][preface]%
  {%
    \noindent\HCode{<h2>\prefacename</h2>}%
  }%
  {%
  }
\else
  \newenvironment{preface}[1][preface]%
  {%
    \jmlrprefaceheader
    \protected@write\@mainauxout
      {}{\string\@prefacestart{\thepage}{\arabic{page}}}%
    \protected@write\@mainauxout{}{\string\@prefacefile{\jmlrprefacepath}{#1}}%
  }%
  {%
    \protected@write\@mainauxout{}{\string\@prefaceend{\thepage}}%
  }
\fi
\newcommand*{\prefacename}{Preface}
\newcommand*{\@prefacefile}[2]{}
\newcommand*{\@prefacestart}[2]{}
\newcommand*{\@prefaceend}[1]{}
\newcommand*{\@prefaceeditor}[1]{}
\newcommand*{\chapterrefname}{Chapter}
\newcommand*{\chaptersrefname}{Chapters}
\newcommand*{\chapterref}[1]{%
  \objectref{#1}{\chapterrefname}{\chaptersrefname}{}{}}
\newcommand*{\articlepageref}[1]{%
  \pageref{#1jmlrstart}%
}
\newcommand*{\articlepagesref}[1]{%
  \pageref{#1jmlrstart}--\pageref{#1jmlrend}%
}
\newcommand*{\@articlepagesref}{%
  \pageref{jmlrstart}--\pageref{jmlrend}%
}
\newcommand*{\articletitleref}[1]{\nameref{#1jmlrstart}}
\newcommand*{\articleauthorref}[1]{%
  \@ifundefined{@jmlr@author@#1}%
  {%
    \ClassWarning{jmlrbook}{Label `#1' undefined}%
  }%
  {%
    \@nameuse{@jmlr@author@#1}%
  }%
}
\renewcommand*\jmlrtitlehook{%
  \hypersetup{pdftitle={\@shorttitle}}%
  \def\xmpTitle{\@shorttitle}%
  \let\jmlrtitlehook\relax
}
\providecommand*{\xmpTitle}{\@title}%
\renewcommand*\jmlrauthorhook{%
  \ifx\@sauthor\@empty
    \hypersetup{pdfauthor={\@author}}%
  \else
    \hypersetup{pdfauthor={\@sauthor}}%
  \fi
  \def\xmpAuthor{\@sauthor}%
  \let\jmlrauthorhook\relax
  \let\@shortauthor\@empty
}
\providecommand*\xmpAuthor{\@author}%
\newcommand*{\@subtitle}{}
\newcommand*{\subtitle}[1]{\renewcommand*{\@subtitle}{#1}}
\newcommand*{\@volume}{\@jmlrvolume}
\newcommand*{\volume}[1]{%
  \renewcommand*{\@volume}{#1}%
  \ifjmlrpdfxa
    \let\xmpVolume\@volume
  \fi
}
\newcommand*{\@issue}{\@jmlrissue}
\newcommand*{\issue}[1]{%
  \renewcommand*{\@issue}{#1}%
  \ifjmlrpdfxa
    \let\xmpIssue\@issue
  \fi
}
\newcommand*{\thejmlrworkshop}{\@jmlrworkshop}
\newcommand*{\@team}{}
\newcommand*{\team}[1]{\renewcommand*{\@team}{#1}}
\newcommand*{\@jmlrlocation}{}
\newcommand*{\jmlrlocation}[1]{%
 \renewcommand*{\@jmlrlocation}{#1}%
 \protected@write\@auxout{}{\string\jmlrbook@location{#1}}%
}
\newcommand*{\@productioneditorname}{Production Editor}
\newcommand*{\@productioneditor}{}
\newcommand*{\productioneditor}[1]{%
  \renewcommand*{\@productioneditor}{#1}%
  \renewcommand*{\@productioneditorname}{Production Editor}%
}
\newcommand*{\productioneditors}[1]{%
  \renewcommand*{\@productioneditor}{#1}%
  \renewcommand*{\@productioneditorname}{Production Editors}%
}
\newcommand*{\@logo}{}
\newcommand*{\logo}[2][]{%
\ifjmlrhtml
  \def\@logo@tmp{#1}%
  \ifx\@logo@tmp\@empty
    \renewcommand*{\@logo}{#2}%
  \else
    \renewcommand*{\@logo}{\HCode{<a href="#1">}#2\HCode{</a>}}%
  \fi
\else
  \renewcommand*{\@logo}{#2}%
\fi
}
\renewcommand*{\booklinebreak}[1][4]{\linebreak[#1]}
\def\c@lbmaketitle{\jmlrmaketitle}
\newcommand*{\maintitle}{}
\newcommand*{\SetTitleElement}[3]{%
  {%
    \expandafter\ifx\csname @#1\endcsname\@empty
    \else
      #2\csname @#1\endcsname#3%
    \fi
  }%
}
\newcommand{\IfTitleElement}[3]{%
  \expandafter\ifx\csname @#1\endcsname\@empty
    #2%
  \else
    #3%
  \fi
}
\newcommand{\titlebody}{%
  \SetTitleElement{title}{\maintitlefont}{\postmaintitle}%
  \SetTitleElement{volume}{\mainvolumefont}{\postmainvolume}%
  \SetTitleElement{subtitle}{\mainsubtitlefont}{\postmainsubtitle}%
  \SetTitleElement{logo}{\mainlogofont}{\postmainlogo}%
  \SetTitleElement{team}{\mainteamfont}{\postmainteam}%
  \SetTitleElement{author}{\mainauthorfont}{\postmainauthor}%
  \SetTitleElement{productioneditor}{\mainproductioneditorfont}%
    {\postmainproductioneditor}%
}
\ifjmlrhtml
  \renewcommand{\c@lamaketitle}{%
    \HCode{<table cellpadding="2" cellspacing="2" border="0" width="100\%">}%
    \HCode{<tbody><tr><td valign="top">}%
    \HCode{<h1>}%
    \@title\newline
    \ifx\@jmlrvolume\@empty
      \ifx\@volume\@empty
      \else
        Volume \@volume
        \ifx\@subtitle\@empty\else: \fi
      \fi
    \else
        Volume \@jmlrvolume
        \ifx\@subtitle\@empty\else: \fi
    \fi
    \@subtitle
    \HCode{</h1>}%
    \newline
    \textbf{Editors: \@author}
    \HCode{</td><td valign="top">}%
    \@logo
    \HCode{</td></tr></tbody></table>}%
    \let\maintitle\@title
  }
\else
  \renewcommand{\c@lamaketitle}{%
    \pagenumbering{alph}%
    \pagestyle{empty}%
    \begin{titlepage}%
      \let\footnotesize\small
      \let\footnoterule\relax
      \let\footnote\thanks
      \titlebody
      \par
      \@thanks
    \end{titlepage}%
    \setcounter{footnote}{0}%
    \let\maintitle\@title
    \c@lmtitlempty
  }
\fi
\renewcommand{\maintitlefont}{%
  \null\vskip15pt\relax\par
  \flushleft\Huge\bfseries\noindent}
\renewcommand{\postmaintitle}{%
  \par\relax
}
\newcommand{\mainvolumefont}{%
  \flushleft\noindent\LARGE\bfseries Volume
}
\newcommand{\postmainvolume}{%
  \IfTitleElement{subtitle}{}{:}\par\relax
}
\newcommand{\mainissuefont}{%
  \flushleft\noindent\LARGE\bfseries Issue
}
\newcommand{\postmainissue}{%
  \par\relax
}
\newcommand{\mainsubtitlefont}{%
  \flushleft\LARGE\bfseries\noindent}
\newcommand{\postmainsubtitle}{\par}
\newcommand{\mainlogofont}{%
  \vfill
  \begin{center}}
\newcommand{\postmainlogo}{\end{center}\vfill\par}
\newcommand{\mainteamfont}{\flushleft\bfseries\Large\noindent}
\newcommand{\postmainteam}{\par}
\renewcommand{\mainauthorfont}{%
  \flushleft\Large\itshape\doublespacing\noindent}
\renewcommand{\postmainauthor}{%
\par}
\newcommand{\mainproductioneditorfont}{%
  \flushleft\Large\noindent \@productioneditorname: \itshape}
\newcommand{\postmainproductioneditor}{\par}
\renewcommand{\maindatefont}{}
\renewcommand{\postmaindate}{}
\ifjmlrhtml
  \newenvironment{signoff}[2][The Editorial Team]{%
    \def\Editor##1{##1\par\vskip\baselineskip\noindent\ignorespaces}%
    \def\@editorialteam{#1}%
    \def\@signoffdate{#2}%
    \par\vskip\baselineskip\noindent
    \ifx\@signoffdate\@empty
    \else
      \emph{\@signoffdate}\nopagebreak\par
      \nopagebreak\vskip\baselineskip\noindent
    \fi
    \ifx\@editorialteam\@empty
    \else
      \@editorialteam:\nopagebreak\par\nopagebreak\vskip\baselineskip
    \fi
    \nopagebreak\noindent\ignorespaces
  }%
  {%
  }%
\else
  \newenvironment{signoff}[2][The Editorial Team]{%
    \def\Editor##1{%
      \protected@write\@mainauxout{}{\string\@prefaceeditor{##1}}%
      \begin{tabular}{@{}l@{}}%
      ##1%
      \end{tabular}%
      \par\vskip\baselineskip\noindent\ignorespaces
    }%
    \def\@editorialteam{#1}%
    \def\@signoffdate{#2}%
    \par\vskip\baselineskip\noindent
    \ifx\@signoffdate\@empty
    \else
      \emph{\@signoffdate}\par
      \vskip\baselineskip\noindent
    \fi
    \ifx\@editorialteam\@empty
    \else
      \@editorialteam:\nopagebreak\par\vskip\baselineskip
    \fi
    \nopagebreak\noindent\ignorespaces
  }%
  {%
  }
\fi
\newenvironment{authorsignoff}{%
  \def\Author##1{\begin{tabular}{@{}p{\linewidth}@{}}%
    ##1%
    \end{tabular}%
    \par\vskip\baselineskip\noindent\ignorespaces
  }%
  \par\vskip\baselineskip\noindent\ignorespaces
}{%
}
\renewcommand{\zeroextracounters}{%
  \@ifundefined{c@theorem}{}{\setcounter{theorem}{0}}%
  \@ifundefined{c@algorithm}{}{\setcounter{algorithm}{0}}%
  \@ifundefined{c@algocf}{}{\setcounter{algocf}{0}}%
  \@ifundefined{c@example}{}{\setcounter{example}{0}}%
  \@ifundefined{c@definition}{}{\setcounter{definition}{0}}%
}
\renewcommand*{\contentsname}{Table of Contents}
\def\theHalgorithm{\theHchapter.\thealgorithm}
\def\theHexample{\theHchapter.\theexample}
\def\theHtheorem{\theHchapter.\thetheorem}
\def\theHsection{\theHchapter.\thesection}
\def\theHsubsection{\theHchapter.\thesubsection}
\def\theHsubsubsection{\theHchapter.\thesubsubsection}
\def\theHparagraph{\theHchapter.\theparagraph}
\def\theHsubfigure{\theHfigure.\arabic{subfigure}}
\def\theHsubtable{\theHtable.\arabic{subtable}}
\def\theHfootnote{\theHchapter.\alpha{footnote}}
\def\theHtable{\theHchapter.\arabic{table}}
\def\theHfigure{\theHchapter.\arabic{figure}}
\def\theHalgocf{\theHchapter.\thealgocf}
\renewcommand*{\mailto}[1]{%
  \href{mailto:#1}{\nolinkurl{#1}}%
}
\c@lhaschapterfalse
\let\c@lthesec\thesection
\newcommand\doimportchapterHref{%
  \edef\@currentHref{chapter.\thechapter}%
}
\def\toclevel@appendix{-1}
\def\Xprefix{}
\DeclareRobustCommand\Xref{\@ifstar\@Xrefstar\T@Xref}%
\DeclareRobustCommand\Xpageref{%
  \@ifstar\@Xpagerefstar\T@Xpageref
}%
\def\HyRef@StarSetXRef#1{%
  \begingroup
    \Hy@safe@activestrue
    \edef\x{#1}%
    \@onelevel@sanitize\x
    \edef\x{\endgroup
      \noexpand\HyRef@@StarSetRef
        \expandafter\noexpand\csname r@\Xprefix\x\endcsname{\x}%
    }%
  \x
}
\def\@Xrefstar#1{%
  \HyRef@StarSetXRef{#1}\@firstoffive
}
\def\@Xpagerefstar#1{%
  \HyRef@StarSetXRef{#1}\@secondoffive
}
\def\T@Xref#1{%
  \Hy@safe@activestrue
  \expandafter\@setXref\csname r@\Xprefix#1\endcsname\@firstoffive{#1}%
  \Hy@safe@activesfalse
}%
\def\T@Xpageref#1{%
  \Hy@safe@activestrue
  \expandafter\@setXref\csname r@\Xprefix#1\endcsname\@secondoffive{#1}%
  \Hy@safe@activesfalse
}%
\def\Xlabel#1{%
  \@bsphack
    \begingroup
      \@onelevel@sanitize\@currentlabelname
      \edef\@currentlabelname{%
        \expandafter\strip@period\@currentlabelname\relax.\relax\@@@
      }%
      \protected@write\@mainauxout{}{%
        \string\newlabel{\Xprefix#1}{{\@currentlabel}{\thepage}%
          {\@currentlabelname}{\@currentHref}{}}%
      }%
    \endgroup
  \@esphack
}
\let\ltx@label\Xlabel
\def\@setXref#1#2#3{% csname, extract group, refname
  \ifx#1\relax
    \protect\G@refundefinedtrue
    \nfss@text{\reset@font\bfseries ??}%
    \@latex@warning{%
      Reference `#3' on page \thepage \space undefined%
    }%
  \else
    \expandafter\Hy@setref@link#1\@empty\@empty\@nil{#2}%
  \fi
}
\AtBeginDocument{%
  \renewcommand\@secondoffive[5]{#2}%
  \jmlrwritepdfinfo
  \let\jmlrwritepdfinfo\relax
}
\def\@setimportlabel{%
  \let\@mainauxout\@auxout
  \let\HRlabel\label
}
\AtBeginDocument{\@jmlrbegindoc}
\newcommand*\@jmlrbegindoc{
  \@setimportlabel
  \gdef\@setimportlabel{\let\ref\Xref \let\pageref\Xpageref}%
  \let\ReadBookmarks\relax
  \@ifundefined{@beginmainauxhook}{}{\@beginmainauxhook}%
}
\let\@org@InputIfFileExists\InputIfFileExists
\newenvironment{jmlrpapers}{%
\def\@begindocumenthook{%
  \@jmlrbegindoc
  \let\bibcite\c@lbNATbibcite
}
\def\@enddocumenthook{%
  \@jmlrenddoc
  \let\bibcite\c@lbNAT@testdef
}
  \begin{papers}[]
  \if@twocolumn
    \def\@jmlr@restore{\twocolumn}%
  \else
    \def\@jmlr@restore{\onecolumn}%
  \fi
  \jmlrarticlecommands
  \let\importpubpaper\@importpubpaper
  \let\importpaper\@importpaper
  \let\importarticle\@importarticle
  \let\label\Xlabel
  \let\ref\Xref
  \pagestyle{article}%
}{%
  \@jmlr@restore
  \end{papers}
}
\newcommand{\addtomaincontents}[2]{%
  \protected@write\@mainauxout{\let\label\@gobble\let\index\@gobble
    \let\glossary\@gobble}{\string\@writefile{#1}{#2}}%
}
\newcommand*{\@write@author}[2]{%
  \def\@jmlr@authors@sep{ and }%
  \protected@write\@mainauxout{}{%
    \string\@new@articleauthor{#1}{#2}%
  }%
}
\newcommand*{\@new@articleauthor}[2]{%
  \expandafter\gdef\csname @jmlr@author@#1\endcsname{%
    \hyperref[#1jmlrstart]{#2}}%
}
\newcommand*{\@@write@jmlr@import}[3]{%
  \protected@write\@mainauxout{}{\string\@jmlr@import{#1}{#2}{#3}}%
}
\newcommand*{\@jmlr@import}[3]{}
\newcommand*{\@@write@jmlr@apdimport}[3]{%
  \protected@write\@mainauxout{}{\string\@jmlr@apdimport{#1}{#2}{#3}}%
}
\newcommand*{\@jmlr@apdimport}[3]{}
\def\@write@jmlr@import{\@@write@jmlr@import}
\def\jmlrpremaketitlehook{%
  \cleardoublepage
  \phantomsection
  \let\@currentlabelname\@shorttitle
  \refstepcounter{chapter}%
}%
\newcommand*{\jmlrimporthook}{}
\newcommand*{\@importpubpaper}[4][\@importdir\@importfile]{%
  \bgroup
    \def\@importdir{#2/}%
    \def\@importfile{#3}%
    \@write@jmlr@import{#1}{#2}{#3}%
    \def\@extra@b@citeb{#1}%
    \def\@extra@binfo{#1}%
    \jmlrpages{#4}%
    \graphicspath{{\@importdir}}%
    \def\jmlrmaketitlehook{%
      \label{}%
      \def\titlebreak{ }%
      \addtomaincontents{toc}%
        {%
          \protect\contentsline{papertitle}{\@title}{\thepage}%
   {page.\thepage}}%
      \pdfbookmark{\@shorttitle}{chapter.\theHchapter}%
      \def\@jmlr@authors@sep{ \& }%
      \tocchapterpubauthor{\@jmlr@authors}%
      {%
        \@jmlrabbrvproceedings
        \ifx\@jmlrvolume\@empty
           \ifx\@jmlrpages\@empty\else\space\fi
        \else
           \space\@jmlrvolume
           \ifx\@jmlrissue\@empty
           \else
              (\@jmlrissue)%
           \fi
           \ifx\@jmlrpages\@empty\else:\fi
        \fi
        \ifx\@jmlrpages\@empty
        \else
           \@jmlrpages
           \ifx\@jmlryear\@empty\else,\fi
        \fi
        \space\@jmlryear
      }%
      \@write@author{#1}{\@jmlr@authors}%
    }%
    \def\InputIfFileExists##1##2##3{%
       \IfFileExists{##1}{%
          \@org@InputIfFileExists{##1}{##2}{##3}%
       }%
       {%
          \@org@InputIfFileExists{\@importdir##1}{##2}{##3}%
       }%
     }%
    \def\Xprefix{#1}%
    \jmlrimporthook
    \import{\@importdir\@importfile}%
    \def\Xprefix{}%
  \egroup
  \gdef\@shortauthor{}%
  \gdef\@shorttitle{}%
  \gdef\@firstauthor{}%
  \gdef\@jmlr@authors{\@jmlrauthors}%
  \gdef\@jmlrauthors{}%
  \gdef\@firstsurname{}%
}
\newcommand{\importpubpaper}[4][]{%
  \ClassError{jmlrbook}{\string\importpubpaper\space
not permitted outside `jmlrpapers' environment}{}%
}
\newcommand{\@importpaper}[3][\@importdir\@importfile]{%
  \bgroup
    \def\@importdir{#2/}%
    \def\@importfile{#3}%
    \@write@jmlr@import{#1}{#2}{#3}%
    \def\@extra@b@citeb{#1}%
    \def\@extra@binfo{#1}%
    \jmlrpages{\protect\@articlepagesref}%
    \graphicspath{{\@importdir}}%
    \def\jmlrmaketitlehook{%
      \label{}%
      \def\titlebreak{ }%
      \addtomaincontents{toc}%
        {%
          \protect\contentsline{papertitle}{\@title}{\thepage}%
   {page.\thepage}}%
      \pdfbookmark{\@shorttitle}{chapter.\theHchapter}%
      \def\@jmlr@authors@sep{ \& }%
      \tocchapterpubauthor{\@jmlr@authors}%
      {%
        \@jmlrabbrvproceedings
        \ifx\@jmlrvolume\@empty
           \space
        \else
           \space\@jmlrvolume
           \ifx\@jmlrissue\@empty
           \else
              (\@jmlrissue)%
           \fi
           :%
        \fi
        \protect\articlepagesref{#1}%
        \ifx\@jmlryear\@empty\else,\fi
        \space\@jmlryear
      }%
      \@write@author{#1}{\@jmlr@authors}%
    }%
    \def\InputIfFileExists##1##2##3{%
       \IfFileExists{##1}{%
          \@org@InputIfFileExists{##1}{##2}{##3}%
       }%
       {%
          \@org@InputIfFileExists{\@importdir##1}{##2}{##3}%
       }%
     }%
    \def\Xprefix{#1}%
    \let\jmlrvolume\@gobble
    \let\jmlryear\@gobble
    \let\jmlrworkshop\@gobble
    \let\jmlrissue\@gobble
    \let\jmlrpages\@gobble
    \jmlrimporthook
    \import{\@importdir\@importfile}%
    \def\Xprefix{}%
  \egroup
  \gdef\@shortauthor{}%
  \gdef\@shorttitle{}%
  \gdef\@firstauthor{}%
  \gdef\@jmlr@authors{\@jmlrauthors}%
  \gdef\@jmlrauthors{}%
  \gdef\@firstsurname{}%
}

\newcommand{\importpaper}[3][]{%
  \ClassError{jmlrbook}{\string\importpaper\space
not permitted outside `jmlrpapers' environment}{}%
}
\newcommand{\@importarticle}[3][\@importdir\@importfile]{%
  \bgroup
    \def\@importdir{#2/}%
    \def\@importfile{#3}%
    \@write@jmlr@import{#1}{#2}{#3}%
    \def\@extra@b@citeb{#1}%
    \def\@extra@binfo{#1}%
    \def\jmlrmaketitlehook{%
    \def\titlebreak{ }%
      \addtomaincontents{toc}%
        {%
          \protect\contentsline{papertitle}{\@title}{\thepage}%
   {page.\thepage}}%
      \label{}%
      \pdfbookmark{\@shorttitle}{chapter.\theHchapter}%
      \def\@jmlr@authors@sep{ \& }%
      \tocchapterauthor{\@jmlr@authors}%
      \@write@author{#1}{\@jmlr@authors}%
      \jmlrpages{}%
      \jmlrvolume{}%
      \jmlryear{}%
      \jmlrsubmitted{}%
      \jmlrpublished{}%
      \jmlrproceedings{}{}%
    }%
    \graphicspath{{\@importdir}}%
    \def\InputIfFileExists##1##2##3{%
       \IfFileExists{##1}{%
          \@org@InputIfFileExists{##1}{##2}{##3}%
       }%
       {%
          \@org@InputIfFileExists{\@importdir##1}{##2}{##3}%
       }%
     }%
    \def\Xprefix{#1}%
    \jmlrimporthook
    \let\ps@jmlrtps\ps@articlet
    \import{\@importdir\@importfile}%
    \def\Xprefix{}%
  \egroup
  \gdef\@shortauthor{}%
  \gdef\@shorttitle{}%
  \gdef\@firstauthor{}%
  \gdef\@jmlr@authors{\@jmlrauthors}%
  \gdef\@jmlrauthors{}%
  \gdef\@firstsurname{}%
}
\newcommand{\importarticle}[3][]{%
  \ClassError{jmlrbook}{\string\importarticle\space
not permitted outside `jmlrpapers' environment}{}%
}
\newcommand{\addtocpart}[1]{%
  \cleardoublepage
  \refstepcounter{tocpart}%
  \addtocontents{toc}{\protect\tocpart{#1}}%
  \pdfbookmark[-1]{#1}{part.\thetocpart}%
}
\newcounter{tocpart}
\newcommand{\tocpart}[1]{%
    \addpenalty{-\@highpenalty}%
    \vskip 1.0ex \@plus\p@
    \setlength\@tempdima{2.25em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \large\bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss \null}\par
      \penalty\@highpenalty
    \endgroup
}
\setlength{\prechapterskip}{3em}
\setlength{\postchapterskip}{20pt}
\renewcommand{\chapternumberformat}[1]{%
 \Large\bfseries \@chapapp\space#1\par
}
\renewcommand{\chaptertitleformat}[1]{%
 \Large\bfseries #1}
\renewcommand*{\chapterformat}{%
   \raggedright
}
\renewcommand{\preparthook}{\cleardoublepage\null\vfil}
\renewcommand{\partnumberformat}[1]{%
  \Huge\bfseries \@partapp\nobreakspace#1\par\nobreak
  \vskip 20\p@
}
\def\postparthook{%
  \thispagestyle{empty}%
  \vfil\newpage
  \null
  \thispagestyle{empty}%
  \newpage
}
\newcommand{\@curparthead}{}
\renewcommand{\parttitleformat}[1]{#1%
  \gdef\@curparthead{\@partapp\space \thepart. #1}%
  \@mkboth{\@curparthead}{\@curparthead}%
}
\renewcommand{\firstpageno}[1]{}
\newcommand{\tocchapterauthor}[1]{%
  \addtomaincontents{toc}{\protect\contentsline{chapterauthor}{%
  #1}{}{}}%
}
\newcommand{\tocchapterpubauthor}[2]{%
  \addtomaincontents{toc}{\protect\contentsline{chapterauthor}{%
  #1; #2.}{}{}}%
}
\renewcommand*\@pnumwidth{2em}
\renewcommand*\l@part[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    %\setlength\@tempdima{5em}%
    \settowidth\@tempdima{\large\bfseries \@partapp\space MM}%
    \vbox{%
      \pagerule
      \begingroup
        \parindent \z@ \rightskip \@pnumwidth
        \parfillskip -\@pnumwidth
        \leavevmode \large\bfseries
        \advance\leftskip\@tempdima
        \hskip -\leftskip
        \renewcommand*\numberline[1]{\hb@xt@ \@tempdima
          {\@partapp\space ##1\hfil }}%
        #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss
           \normalfont\normalsize #2}\par
        \penalty\@highpenalty
      \endgroup
      \pagerule
    }%
  \fi}
\renewcommand{\l@chapter}[2]{%
  \ifnum\c@tocdepth>\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus \p@
    \setlength\@tempdima{2em}%
    \begingroup
      \parindent \z@
      \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \large \bfseries
      \advance \leftskip \@tempdima
      \hskip -\leftskip
        \renewcommand*\numberline[1]{\hb@xt@ \@tempdima
          {##1\hfil }}%
      #1\nobreak \hfil \nobreak \hb@xt@ \@pnumwidth {\hss
       \normalfont\normalsize #2}\par
      \penalty \@highpenalty
    \endgroup
  \fi
}
\newcommand*{\l@papertitle}[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{3em}%
    \begingroup
      \leavevmode \raggedright\itshape
      #1\nobreak\hfill \nobreak\hb@xt@\@pnumwidth{\hss
       \normalfont#2}%
       \par
      \penalty\@highpenalty
    \endgroup
  \fi
}
\newcommand*\l@chapterauthor[2]{%
  \ifnum \c@tocdepth >\m@ne
    \begingroup
      \parindent \z@
      \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \raggedright
      \parbox{\linewidth-\@pnumwidth}{\raggedright#1\par}%
       \par
    \endgroup
  \fi}
\renewcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{3em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \normalsize\mdseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\renewcommand*\l@subsection[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{3.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \normalsize\mdseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\renewcommand*{\chaptermark}[1]{%
  \@mkboth{\@curparthead}{\protect\thechapter. #1}%
}
\newcommand{\firstpagehead}{}
\newcommand{\firstpagefoot}{%
  \@reprint\hfill\thepage
}
\newcommand*{\headfont}{\reset@font\small\scshape}%
\newcommand*{\footfont}{\reset@font\small\itshape}%
\newcommand*{\ps@chplain}{%
  \let\@mkboth\@gobbletwo
  \renewcommand*{\@oddhead}{\headfont\firstpagehead}%
  \renewcommand*{\@evenhead}{}%
  \renewcommand*{\@oddfoot}{\footfont\firstpagefoot}%
  \renewcommand*{\@evenfoot}{\footfont\thepage\hfill
  }%
}
\let\ps@plain\ps@chplain
\newcommand*{\ps@article}{%
  \let\@mkboth\@gobbletwo
  \renewcommand*{\@oddhead}{\headfont\hfill\@shorttitle}%
  \renewcommand*{\@evenhead}{\headfont\@shortauthor\hfill}%
  \renewcommand*{\@oddfoot}{\footfont\hfill\thepage}
  \renewcommand*{\@evenfoot}{\footfont\thepage\hfill}
}
\newcommand*{\ps@articlet}{%
  \let\@mkboth\@gobbletwo
  \renewcommand*{\@oddhead}{}%
  \renewcommand*{\@evenhead}{}%
  \renewcommand*{\@oddfoot}{\footfont\hfill\thepage}
  \renewcommand*{\@evenfoot}{\footfont\thepage\hfill}
}
\newcommand*{\ps@jmlrbook}{%
  \renewcommand*{\@oddfoot}{\footfont\hfill\thepage}
  \renewcommand*{\@evenfoot}{\footfont\thepage\hfill}
  \def\@evenhead{\headfont\leftmark\hfill}%
  \def\@oddhead{\hfill\headfont\rightmark}%
  \let\@mkboth\markboth
  \renewcommand*{\sectionmark}[1]{}%
}
\newcommand*{\markleft}[1]{%
  \begingroup
    \let\label\relax
    \let\index\relax
    \let\glossary\relax
    \expandafter\@markleft\@themark{#1}%
    \@temptokena
    \expandafter{\@themark}%
    \mark{\the\@temptokena}
  \endgroup
  \if@nobreak
    \ifvmode
      \nobreak
    \fi
  \fi
}
\newcommand*{\@markleft}[3]{%
  \@temptokena{#2}%
  \unrestored@protected@xdef\@themark{{#3}{\the\@temptokena}}
}
\renewcommand*{\morefrontmatter}{\pagestyle{jmlrbook}%
  \def\chaptermark##1{%
    \@mkboth{##1\hfill}{\hfill##1}}%
}
\renewcommand*{\moremainmatter}{\pagestyle{jmlrbook}%
  \def\chaptermark##1{%
    \@mkboth{\@curparthead}{\protect\thechapter. ##1}%
  }%
}
\renewcommand*\bibsection{\section*{\refname}}
\jmlrbookcommands
\providecommand*{\SetNoLine}{\SetAlgoNoLine}
\providecommand*{\SetVline}{\SetAlgoVlined}
\providecommand*{\Setvlineskip}{\SetVlineSkip}
\providecommand*{\SetLine}{\SetAlgoLined}
\providecommand*{\dontprintsemicolon}{\DontPrintSemicolon}
\providecommand*{\printsemicolon}{\PrintSemicolon}
\providecommand*{\incmargin}{\IncMargin}
\providecommand*{\decmargin}[1]{\DecMargin{-#1}}
\providecommand*{\setnlskip}{\SetNlSkip}
\providecommand*{\Setnlskip}{\SetNlSkip}
\providecommand*{\setalcapskip}{\SetAlCapSkip}
\providecommand*{\setalcaphskip}{\SetAlCapHSkip}
\providecommand*{\nlSty}{\NlSty}
\providecommand*{\Setnlsty}{\SetNlSty}
\providecommand*{\linesnumbered}{\LinesNumbered}
\providecommand*{\linesnotnumbered}{\LinesNotNumbered}
\providecommand*{\linesnumberedhidden}{\LinesNumberedHidden}
\providecommand*{\showln}{\ShowLn}
\providecommand*{\showlnlabel}{\ShowLnLabel}
\providecommand*{\nocaptionofalgo}{\NoCaptionOfAlgo}
\providecommand*{\restorecaptionofalgo}{\RestoreCaptionOfAlgo}
\providecommand*{\restylealgo}{\RestyleAlgo}
\providecommand*{\Titleofalgo}{\TitleOfAlgo}
\endinput
%%
%% End of file `jmlrbook.cls'.
