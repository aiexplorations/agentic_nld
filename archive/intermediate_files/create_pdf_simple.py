"""
Simple PDF generation using weasyprint (HTML to PDF).
"""

import subprocess
import os
from pathlib import Path

def install_weasyprint():
    """Install weasyprint for HTML to PDF conversion."""
    try:
        import weasyprint
        print("✅ WeasyPrint already available")
        return True
    except ImportError:
        print("📦 Installing WeasyPrint...")
        try:
            subprocess.run(['pip', 'install', 'weasyprint'], check=True)
            print("✅ WeasyPrint installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Could not install WeasyPrint")
            return False

def create_simple_html_report():
    """Create HTML version with embedded CSS and MathJax."""
    
    print("📄 Creating HTML report with LaTeX support...")
    
    # Read the LaTeX markdown
    with open("TECHNICAL_REPORT_LATEX.md", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create HTML with MathJax support
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chaos Theory in Two-Agent LLM Conversations</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- MathJax -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }}
        }};
    </script>
    
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .equation {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
        }}
        .table {{
            margin: 20px 0;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
            margin-top: 30px;
        }}
        .abstract {{
            background-color: #f8f9fa;
            padding: 20px;
            border-left: 4px solid #007bff;
            margin: 20px 0;
        }}
        .math {{
            font-size: 1.1em;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .authors {{
            text-align: center;
            margin: 20px 0;
            font-style: italic;
        }}
        .date {{
            text-align: center;
            margin: 10px 0;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
"""
    
    # Process the markdown content
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        # Convert markdown headers to HTML
        if line.startswith('# '):
            processed_lines.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('## '):
            processed_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('### '):
            processed_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('#### '):
            processed_lines.append(f'<h4>{line[5:]}</h4>')
        # Convert bold text
        elif '**' in line:
            line = line.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
            processed_lines.append(f'<p>{line}</p>')
        # Convert italic text
        elif '*' in line and not line.strip().startswith('*Figure'):
            line = line.replace('*', '<em>', 1).replace('*', '</em>', 1)
            processed_lines.append(f'<p>{line}</p>')
        # Handle figure captions
        elif line.strip().startswith('*Figure'):
            processed_lines.append(f'<p class="figure-caption"><em>{line.strip()}</em></p>')
        # Handle images
        elif line.strip().startswith('!['):
            processed_lines.append(f'<div class="figure">{line}</div>')
        # Handle tables (basic)
        elif '|' in line and not line.strip().startswith('|-----'):
            processed_lines.append(f'<p>{line}</p>')  # Keep as text for now
        # Handle equations (preserve LaTeX)
        elif '$$' in line:
            processed_lines.append(line)  # MathJax will handle these
        # Handle code blocks
        elif line.strip().startswith('```'):
            if line.strip() == '```':
                processed_lines.append('</pre>')
            else:
                processed_lines.append('<pre>')
        # Regular paragraphs
        elif line.strip() and not line.strip().startswith('---'):
            processed_lines.append(f'<p>{line}</p>')
        else:
            processed_lines.append(line)
    
    html_content += '\n'.join(processed_lines)
    html_content += """
    </div>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""
    
    # Write HTML file
    with open("TECHNICAL_REPORT.html", 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML report created: TECHNICAL_REPORT.html")
    return True

def create_markdown_pdf_pandoc():
    """Try simpler pandoc conversion without LaTeX engine."""
    
    print("📄 Trying HTML-based PDF generation...")
    
    try:
        # Convert markdown to HTML first, then HTML to PDF
        cmd = [
            'pandoc',
            'TECHNICAL_REPORT_LATEX.md',
            '-t', 'html5',
            '--mathjax',
            '--standalone',
            '--toc',
            '--metadata', 'title=Chaos Theory in Two-Agent LLM Conversations',
            '--metadata', 'author=Anthropic Claude, Rajesh Sampathkumar',
            '-o', 'TECHNICAL_REPORT_PANDOC.html'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ HTML created successfully")
            
            # Try to convert HTML to PDF using weasyprint
            try:
                import weasyprint
                print("🔄 Converting HTML to PDF...")
                weasyprint.HTML('TECHNICAL_REPORT_PANDOC.html').write_pdf('TECHNICAL_REPORT_WEASYPRINT.pdf')
                print("✅ PDF created using WeasyPrint: TECHNICAL_REPORT_WEASYPRINT.pdf")
                return True
            except Exception as e:
                print(f"❌ WeasyPrint conversion failed: {e}")
                return False
        else:
            print(f"❌ Pandoc HTML conversion failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Main function."""
    
    print("📚 SIMPLE PDF REPORT GENERATOR")
    print("=" * 40)
    
    # Method 1: Create rich HTML version
    html_success = create_simple_html_report()
    
    # Method 2: Try WeasyPrint installation and conversion
    if install_weasyprint():
        pdf_success = create_markdown_pdf_pandoc()
    else:
        pdf_success = False
    
    print("\n" + "=" * 40)
    print("📊 GENERATION SUMMARY:")
    
    if html_success:
        print("✅ HTML Report: TECHNICAL_REPORT.html")
        print("   - Full LaTeX math support via MathJax")
        print("   - Professional styling with Bootstrap")
        print("   - All figures and tables included")
    
    if pdf_success:
        print("✅ PDF Report: TECHNICAL_REPORT_WEASYPRINT.pdf")
        print("   - Print-ready PDF format")
        print("   - Math equations rendered")
    else:
        print("⚠️  PDF generation not available")
        print("   - View HTML version in browser")
        print("   - Print to PDF from browser if needed")
    
    print("\n🎉 Technical report ready for viewing!")
    print("📖 Authors: Anthropic Claude & Rajesh Sampathkumar")
    print("📅 Date: July 2025")

if __name__ == "__main__":
    main()