%do not modify
% Created 23 September 2019 (Eyob Ghebreiesus)
%
% Updated 17 October 2019      
%   • Modified margins to have one inch all around
%   • Added year to titles and such. Just use \setyear{2019-20} and it will add that anywhere a year is needed
%   • Modified the header so that either the school name or logo will appear. Use \setlogo[LogoHeight]{LogoName.png} to specify the logo image to use. If no logo is specified, the school name will be used instead. Optionally, the logo height may be specified.
%   • Table of contents has black text and references have iitred text.
%   • Made small formatting changes
%
% Updated 22 October 2019    
%   • The custom tables were only working in this file. It was a problem with tabu not wanting to work with \columncolor. So I changed it so that tabu is not required and it uses a tabular. It's more complex and you need to use custom column definitions.
%
% Updated 3 November 2019  
%   • Defined a way to change the title page text color. Use \titlecolor to define the color. The argument provided will go into a \color in the \maketitle function
%
% Updated 26 April 2020       
%   • Added contour package to outline text. Used primarily to add outline to cover text so that it is easier seen on images.
%   • Changed the look of the \sqrt symbol to have a closing bar.
%
% Updated 4 October 2020     
%   • Added variables environment. This allows for the meaning of variables to be easily defined. Example usage:
%       \begin{equation}
%           C_L = \frac{2L}{\rho v^2S}
%       \end{equation}
%       \begin{variables}
%           C_L & Lift coefficient\\
%           L & Lift [lbf]\\
%           \rho & Air density [slug/ft$^3$]\\
%           v & Free stream velocity [ft/s]\\
%           S & Wing planform area [ft$^2$]\\
%       \end{variables}
% Updated 11 December 2021      (Eyob )
%   • Added new bibtex.bib file for cites and references
%   • Moded the contents page for auto hyperlinks. 
% Updated 23 May 2022      (Eyob )
%   • New cover and custom logo header files. 
%   • Generated List of Tables and Figures. This will make the file generate all list of figures and table          automatically.
%   • New acknowledgment page, Homepage and Git syncs. Also serves us the authors page for all. 
% Updated 21 August 2022      (Eyob )
%   • \pagenumbering{gobble}, \usetikzlibrary{shapes, arrows} packaging moded
%   • updated guide automatically.
%   • Coloring table with ticks and individual cell shapes.

% Updated 10 September 2022      (Eyob )
%   •  Added new counter tags for subsections titles to show up in the contents page depending on the table of contents. You can comment out in the main page to not show the subsubsection title. 
%   \setcounter{tocdepth}{3}  
%   \setcounter{secnumdepth}{3}

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{aiaa}[2018/09/23 Format for DBF reports]

% Load base class and required packages
\LoadClass[12pt]{article}
\RequirePackage[utf8]{inputenc}         % Specify character input
\RequirePackage[paperwidth=8.5in,paperheight=11in,margin=0.8in,headheight=4in,headsep=8pt]{geometry}   % Set margin. Also resize specific pages
\RequirePackage{setspace}   % Set line spacing
\RequirePackage{mathtools}  % Do fun math stuff
\RequirePackage{graphicx}   % Show images
\RequirePackage{float}      % Position images, tables, etc.
\RequirePackage{subcaption} % Multiple pictures/tables/etc.
\RequirePackage{wrapfig}    % Wrap image around text
\RequirePackage[table]{xcolor}  % Add colors
\RequirePackage{siunitx}    % Set how many decimal places to show
\RequirePackage{multirow}   % Allow for tables with cells that take up multiple columns/rows
\RequirePackage{tikz}       % Make plots from csv files
\RequirePackage{pgfplots}   % Make tables from csv files
\RequirePackage{fancyhdr}   % Make header and footer nice
\RequirePackage{graphbox}   % Align image and text in header
\RequirePackage{eso-pic}    % Place cover image in background
%\RequirePackage{tabu}       % Do fancy table stuff
\RequirePackage{titlesec}   % Manually set section title spacing
\RequirePackage[RPvoltages]{circuitikz} % Draw circuits
\RequirePackage{tocloft}    % ToC spacing
\RequirePackage{cancel}     % Make things go to zero (or other things) in equations
\RequirePackage{hyperref}   % Make stuff clickable
\RequirePackage{pgfplotstable}  % Turn CSV into a table
\RequirePackage[bottom]{footmisc}   % Keep footnote at bottom of page
\RequirePackage{enumitem}   % Make lists nicer
\RequirePackage{boldline}   % Make thicker hlines
\RequirePackage{array}      % Do spacing things
\RequirePackage{pdfpages}   % Include PDF pages
\RequirePackage{contour}    % At outline to text
\RequirePackage{colortbl}	% Coloring tables
\RequirePackage{bm}			% Bold math symbols
\RequirePackage[cm]{sfmath}	% Use sans-serif font for math
\RequirePackage{amssymb}	% Additional math symbols

% Random compatibility thing
\pgfplotsset{compat=1.15}

% Variables
% Set the type of report
\newcommand{\settype}[1]{\def \type{#1}}

% Set the cover image
\newcommand{\coverimage}[1]{\def \cover{#1}}

% Set the competition year
\newcommand{\setyear}[1]{\def \year{#1}}

% Set logo to use in header
\newcommand{\setlogo}[2][13pt]{\def \logo{#2} \def \logoheight{#1}}

% Set title page text color
\def \Tcolor{white}
\newcommand{\titlecolor}[1]{\def \Tcolor{#1}}

% Set image folder
\graphicspath{{Images/}{../Images/}}
%----------


% Declare document options
\DeclareOption{twocolumn}{\OptionNotUsed}
\DeclareOption{proposal}{\settype{Proposal}\def\proposal}
\DeclareOption{report}{\settype{Design Report}\def\report}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
%----------



% Make title page
\renewcommand{\maketitle}{
%
    % Report title
    \ifdefined\report
        % Set contour length
        \contourlength{1.5pt}
    
        % Remove header and footer
        \thispagestyle{empty}
        
        % Show background image of the report
        \ifdefined\cover
            \AddBackground{\cover}
        \fi
        
        % Title text
        \begin{center}
            %\vspace*{\fill}
            \color{\Tcolor}
            \contour{black}{\Huge \year ~AIAA Design/Build/Fly \type} \par
            \setlength{\parskip}{.6em}
            \contour{black}{\LARGE Illinois Institute of Technology}\par
            \vspace*{\fill}
        \end{center}
        
        \pagenumbering{gobble}
        
        % Add a page after title
        \newpage    
        
        % Start showing page numbers
        \pagenumbering{arabic}
        \setcounter{page}{2}
        
        % Show table of contents for report
        {
            \hypersetup{linkcolor=black}
            \tableofcontents
            \listoffigures
            \listoftables
        }
        
        \newpage
        
        \renewcommand \thesection{\Roman{section}}
        \setcounter{secnumdepth}{0}
        \section{Abbreviations, Acronyms, and Symbols}
        
        %
        \renewcommand \thesection{\arabic{section}}
        \setcounter{secnumdepth}{3}
    \fi
% End report title
%
% Proposal title
    \ifdefined\proposal
        \begin{center}
            \vspace{.5em}
            \huge \year ~AIAA Design, Build, Fly \type
            \vspace{-.75em}
        \end{center}
    \fi
}
%----------



% Set font to Arial and 1.5 line spacing
\renewcommand{\rmdefault}{phv}
\renewcommand{\sfdefault}{phv}
\onehalfspacing
%----------



% Define IIT colors
% These are the official school colors according to IIT graphic standards (https://web.iit.edu/marketing-communications/resources)
\definecolor{iitred}{cmyk}{0.12, 1, 0.91, 0.03}
\definecolor{iitgray}{cmyk}{0.56, 0.46, 0.44, 0.1}
\definecolor{iitcyan}{cmyk}{0.56, 0.46, 0.44, 0.1}
\definecolor{iitmist}{cmyk}{0.94, 1.0, 1.0}
\definecolor{iitblue}{cmyk}{0.74, 0.83, 0.9}
\definecolor{iitalmond}{cmyk}{0.94, 0.87, 0.8}
%----------
% Make table of contents clickable
\hypersetup{
    colorlinks,
    citecolor=iitred,
    linkcolor=iitred,
    urlcolor=iitred
}
%----------
% Define header and footer style
% Tell latex to use our custom header and footer
\pagestyle{fancy}
\fancyhf{}
% Height of header
% \setlength{\headheight}{24pt}
% Add red line below header and above footer
\renewcommand{\headrule}{
    \color{gray}\rule{\linewidth}{1pt}
}
\renewcommand{\footrule}{
    \color{gray}\rule{\linewidth}{1pt}
}

% Define the parts of the header
\lhead{
    \ifdefined\logo
        \includegraphics[height=\logoheight, keepaspectratio, align=c]{\logo}
    \else
        Illinois Institute of Technology
    \fi
}
    
%\lhead{Illinois Institute of Technology}
\rhead{\year \type}

% Define the parts of the footer
\cfoot{\raisebox{-2\height}{\color{black}\thepage}}
%----------



% Add background image to title page
\newcommand{\AddBackground}[1]{
    \AddToShipoutPicture*{
        \put(-3,0){
            \parbox[b][\paperheight]{\paperwidth}{
                \vfill
                \centering
                \includegraphics[width = \paperwidth, height = \paperheight]{#1}%, keepaspectratio]{#1}
                \vfill
            }
        }
    }
}
%----------



% Define document settings
\setcounter{tocdepth}{2}
\renewcommand\thesection{\arabic{section}}
\sisetup{
    round-mode = places,    % Round
    round-precision = 2,    % to 2 decimal places
}
\setlength{\parskip}{.75em}
\setlength{\parindent}{0pt}

\titlespacing*{\section}{0pt}{0pt}{-5pt}
\titlespacing*{\subsection}{0pt}{1pt}{-4pt}
\titlespacing*{\subsubsection}{0pt}{3pt}{-5pt}

\setlength{\intextsep}{0pt}
\setlength{\belowcaptionskip}{-5pt}

\numberwithin{equation}{section}
\numberwithin{figure}{section}
\numberwithin{table}{section}

\captionsetup{labelsep = endash}
\captionsetup{labelfont = bf}
%----------



% Redefine these functions because they annoy me
\let\mr\multirow
\let\mc\multicolumn

% Define custom lines
\newcommand{\thickline}{\tabucline[1pt]{-}}
\newcommand{\thicklineb}{\tabucline[1pt]{2-}}

% Add a bulleted item to a table
\newcommand{\tabitem}{~\llap{\textbullet}~~}

% Add a closing bar to the \sqrt symbol
\let\oldsqrt\sqrt
\def\sqrt{\mathpalette\DHLhksqrt}
   \def\DHLhksqrt#1#2{
   \setbox0=\hbox{$#1\oldsqrt{#2\,}$}\dimen0=\ht0
   \advance\dimen0-0.2\ht0
   \setbox2=\hbox{\vrule height\ht0 depth -\dimen0}
   {\box0\lower0.4pt\box2}}
   
% Create new environment for equation variable definitions
\newenvironment{variables}
{
    \begin{center}
        \vspace{-.75em}
        \begin{tabular}{>{$}r<{$} @{~$=$~} >{\raggedright\arraybackslash}l}
            \text{Where:}\qquad
}
{
        \end{tabular}
        \vspace{-.5em}
    \end{center}
}
%----------


% Define table environment to simplify formatting

% New code to make a row have a specific color
\makeatletter
\newcommand*{\@rowstyle}{}
% Sets the style of the next row
\newcommand*{\rowstyle}[1]{\gdef\@rowstyle{#1}\@rowstyle\ignorespaces}
% Resets the row style
\newcolumntype{=}{>{\gdef\@rowstyle{}}}
\newcolumntype{+}{% adds the current row style to the next column
  >{\@rowstyle}%
}
\makeatother

% Define new columns to use the colors
\newcolumntype{L}{+l}
\newcolumntype{C}{+c}
\newcolumntype{R}{+r}

% Table with red first line
\newenvironment{at}[4][H]{
    \begin{table}[#1]
        \centering
        \setlength{\abovecaptionskip}{0pt}
        \setlength{\belowcaptionskip}{5pt}
        \caption{#2}
        \label{#3}
        \begin{tabular}{=#4}
            \rowcolor{iitred}
        \rowstyle{\color{white}}
}{
        \end{tabular}
    \end{table}
}

% Table with red first line and gray first column
\newenvironment{atb}[4][H]{
    \begin{table}[#1]
        \centering
        \setlength{\abovecaptionskip}{0pt}
        \setlength{\belowcaptionskip}{5pt}
        \caption{#2}
        \label{#3}
        \begin{tabular}{>{\color{white}\columncolor{iitgray}}=#4}
            \rowcolor{iitred}
            \rowstyle{\color{white}}
}{
        \end{tabular}
    \end{table}
}

% Table with gray!50 first row and black cell text in first column
\newenvironment{atg}[4][H]{
    \begin{table}[#1]
        \centering
        \setlength{\abovecaptionskip}{0pt}
        \setlength{\belowcaptionskip}{5pt}
        \caption{#2}
        \label{#3}
        \begin{tabular}{>{\color{black}\columncolor{iitgray}}=#4}
            \rowcolor{gray!50}
            \rowstyle{\color{white}}
}{
        \end{tabular}
    \end{table}
}

\newenvironment{atn}[4][H]{
    \begin{table}[#1]
        \centering
        \setlength{\abovecaptionskip}{0pt}
        \setlength{\belowcaptionskip}{5pt}
        \caption{#2}
        \label{#3}
        \begin{tabular}{>{\color{black}\columncolor{white}}=#4}
            \rowcolor{gray!50}
            \rowstyle{\color{white}}
}{
        \end{tabular}
    \end{table}
}

% Use the following if tabu wants to cooperate (\rowfont not working). They're nicer to use but don't work as of October 2019
%\newenvironment{at}[4][H]{
%    \begin{table}[#1]
%        \centering
%        \setlength{\abovecaptionskip}{0pt}
%        \setlength{\belowcaptionskip}{5pt}
%        \caption{#2}
%        \label{#3}
%        \begin{tabu}{#4}
%            \rowfont{\color{white}}
%            \rowcolor{iitred}
%}{
%        \end{tabu}
%    \end{table}
%}
%
%\newenvironment{atb}[4][H]{
%    \begin{table}[#1]
%        \centering
%        \setlength{\abovecaptionskip}{0pt}
%        \setlength{\belowcaptionskip}{5pt}
%        \caption{#2}
%        \label{#3}
%        \begin{tabu}{>{\color{white}\columncolor{iitgray}}#4}
%            \rowfont{\color{white}}
%            \rowcolor{iitred}
%}{
%        \end{tabu}
%    \end{table}
%}
%----------



% Allow for jumping across wires in circuits
\tikzset{
    declare function={% in case of CVS which switches the arguments of atan2
        atan3(\a,\b)=ifthenelse(atan2(0,1)==90, atan2(\a,\b), atan2(\b,\a));},
    %
    kinky cross radius/.initial=+.125cm,
    @kinky cross/.initial=+,
    kinky crosses/.is choice,
    kinky crosses/left/.style={@kinky cross=-},
    kinky crosses/right/.style={@kinky cross=+},
    kinky cross/.style args={(#1)--(#2)}{
        to path={
            let \p{@kc@}=($(\tikztotarget)-(\tikztostart)$),
            \n{@kc@}={atan3(\p{@kc@})+180} in %
            -- ($(intersection of \tikztostart %
            --{\tikztotarget} and #1--#2)! %
            \pgfkeysvalueof{/tikz/kinky cross radius}! %
            (\tikztostart)$) %
            arc [radius=\pgfkeysvalueof{/tikz/kinky cross radius},
            start angle=\n{@kc@},
            delta angle=\pgfkeysvalueof{/tikz/@kinky cross}180]
            -- (\tikztotarget)
        }
    }
}
%----------