"""
Generate PDF version of the technical report using pandoc.
"""

import subprocess
import os
from pathlib import Path

def generate_pdf_report():
    """Generate PDF from LaTeX markdown using pandoc."""
    
    print("📄 Generating PDF Technical Report...")
    
    # Check if pandoc is installed
    try:
        result = subprocess.run(['pandoc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
        print("✅ Pandoc found")
    except FileNotFoundError:
        print("❌ Pandoc not found. Installing via brew...")
        try:
            subprocess.run(['brew', 'install', 'pandoc'], check=True)
            print("✅ Pandoc installed successfully")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Could not install pandoc. Please install manually:")
            print("   macOS: brew install pandoc")
            print("   Linux: sudo apt-get install pandoc")
            print("   Windows: Download from https://pandoc.org/installing.html")
            return False
    
    # Check if LaTeX is available (for PDF generation)
    try:
        result = subprocess.run(['pdflatex', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
        print("✅ LaTeX found")
    except FileNotFoundError:
        print("⚠️  LaTeX not found. Installing BasicTeX...")
        try:
            subprocess.run(['brew', 'install', '--cask', 'basictex'], check=True)
            print("✅ BasicTeX installed. You may need to restart terminal.")
            print("   Also run: sudo tlmgr update --self && sudo tlmgr install collection-fontsrecommended")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Could not install LaTeX. Please install manually:")
            print("   macOS: brew install --cask basictex")
            print("   Linux: sudo apt-get install texlive-latex-base texlive-latex-recommended")
            print("   Windows: Install MiKTeX from https://miktex.org/")
            return False
    
    # Create pandoc command
    input_file = "TECHNICAL_REPORT_LATEX.md"
    output_file = "TECHNICAL_REPORT.pdf"
    
    # Pandoc command with LaTeX options
    pandoc_cmd = [
        'pandoc',
        input_file,
        '-o', output_file,
        '--pdf-engine=pdflatex',
        '--template=eisvogel',  # Professional template if available
        '--listings',  # Code syntax highlighting
        '--number-sections',  # Number sections
        '--toc',  # Table of contents
        '--toc-depth=3',
        '--metadata', 'title=Chaos Theory in Two-Agent LLM Conversations',
        '--metadata', 'author=Anthropic Claude, Rajesh Sampathkumar',
        '--metadata', 'date=July 2025',
        '--metadata', 'geometry=margin=1in',
        '--metadata', 'fontsize=11pt',
        '--metadata', 'documentclass=article',
        '--metadata', 'classoption=twocolumn',  # Two-column layout like academic papers
        '--filter', 'pandoc-citeproc',  # Bibliography processing
        '-V', 'linkcolor=blue',
        '-V', 'urlcolor=blue',
        '-V', 'toccolor=blue'
    ]
    
    # Try with eisvogel template first, fallback to default
    try:
        print("🔄 Converting markdown to PDF with professional template...")
        result = subprocess.run(pandoc_cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode != 0:
            # Try without eisvogel template
            print("⚠️  Professional template not found, using default...")
            pandoc_cmd_simple = [
                'pandoc',
                input_file,
                '-o', output_file,
                '--pdf-engine=pdflatex',
                '--listings',
                '--number-sections',
                '--toc',
                '--toc-depth=3',
                '--metadata', 'title=Chaos Theory in Two-Agent LLM Conversations',
                '--metadata', 'author=Anthropic Claude, Rajesh Sampathkumar',
                '--metadata', 'date=July 2025',
                '--metadata', 'geometry=margin=1in',
                '--metadata', 'fontsize=11pt',
                '-V', 'linkcolor=blue',
                '-V', 'urlcolor=blue'
            ]
            
            result = subprocess.run(pandoc_cmd_simple, capture_output=True, text=True, cwd='.')
            
        if result.returncode == 0:
            print(f"✅ PDF generated successfully: {output_file}")
            
            # Check file size
            if Path(output_file).exists():
                size_mb = Path(output_file).stat().st_size / (1024 * 1024)
                print(f"📊 PDF size: {size_mb:.1f} MB")
            
            return True
        else:
            print(f"❌ Error generating PDF:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during PDF generation: {e}")
        return False

def create_html_version():
    """Create HTML version as fallback."""
    
    print("\n📄 Generating HTML version as backup...")
    
    input_file = "TECHNICAL_REPORT_LATEX.md"
    output_file = "TECHNICAL_REPORT.html"
    
    pandoc_cmd = [
        'pandoc',
        input_file,
        '-o', output_file,
        '--standalone',
        '--mathjax',  # MathJax for equation rendering
        '--highlight-style=github',  # Code highlighting
        '--toc',
        '--toc-depth=3',
        '--metadata', 'title=Chaos Theory in Two-Agent LLM Conversations',
        '--metadata', 'author=Anthropic Claude, Rajesh Sampathkumar',
        '--metadata', 'date=July 2025',
        '--css=https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
        '-V', 'geometry=margin=1in'
    ]
    
    try:
        result = subprocess.run(pandoc_cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print(f"✅ HTML generated successfully: {output_file}")
            return True
        else:
            print(f"❌ Error generating HTML:")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during HTML generation: {e}")
        return False

def main():
    """Main function to generate report in multiple formats."""
    
    print("📚 TECHNICAL REPORT GENERATOR")
    print("=" * 40)
    
    # Check if input file exists
    if not Path("TECHNICAL_REPORT_LATEX.md").exists():
        print("❌ TECHNICAL_REPORT_LATEX.md not found!")
        return
    
    # Try to generate PDF
    pdf_success = generate_pdf_report()
    
    # Generate HTML as backup
    html_success = create_html_version()
    
    print("\n" + "=" * 40)
    if pdf_success:
        print("✅ PDF report generated successfully!")
        print("📄 File: TECHNICAL_REPORT.pdf")
    else:
        print("❌ PDF generation failed")
        
    if html_success:
        print("✅ HTML report generated successfully!")  
        print("🌐 File: TECHNICAL_REPORT.html")
    else:
        print("❌ HTML generation failed")
    
    if pdf_success or html_success:
        print("\n🎉 Report generation completed!")
        print("📊 The report includes:")
        print("   - Complete mathematical framework with LaTeX equations")
        print("   - Comprehensive experimental results")
        print("   - Statistical analysis and hypothesis testing")
        print("   - High-quality figures and visualizations")
        print("   - Full appendices with implementation details")
    else:
        print("\n❌ Report generation failed!")
        print("💡 Alternative: View TECHNICAL_REPORT_LATEX.md directly")
        print("   Most markdown viewers support LaTeX math rendering")

if __name__ == "__main__":
    main()