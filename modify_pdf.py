import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
import io

def modify_pdf(input_pdf, output_pdf):
    """
    Create a modified version of the PDF with new name, roll number, and subgroup
    """
    # Read the original PDF
    pdf_reader = PyPDF2.PdfReader(input_pdf)
    pdf_writer = PyPDF2.PdfWriter()
    
    # Get the first page to overlay
    original_page = pdf_reader.pages[0]
    
    # Create a new PDF with white rectangles to cover old text and new text
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=A4)
    
    # Set font
    can.setFont("Helvetica-Bold", 12)
    
    # Cover the old name with white rectangle and write new name
    # Adjust coordinates based on where the name appears in your PDF
    can.setFillColor(colors.white)
    can.rect(100, 750, 400, 30, fill=1, stroke=0)  # Cover old name
    
    can.setFillColor(colors.black)
    can.drawString(110, 760, "Name: Harshit Sharma")
    
    # Cover old roll number and write new one
    can.setFillColor(colors.white)
    can.rect(100, 720, 400, 25, fill=1, stroke=0)  # Cover old roll no
    
    can.setFillColor(colors.black)
    can.drawString(110, 730, "Roll No: 102216014")
    
    # Cover old subgroup and write new one
    can.setFillColor(colors.white)
    can.rect(100, 690, 400, 25, fill=1, stroke=0)  # Cover old subgroup
    
    can.setFillColor(colors.black)
    can.drawString(110, 700, "Subgroup: 4Q24")
    
    can.save()
    
    # Move to the beginning of the BytesIO buffer
    packet.seek(0)
    overlay_pdf = PyPDF2.PdfReader(packet)
    
    # Merge the overlay with the first page
    original_page.merge_page(overlay_pdf.pages[0])
    pdf_writer.add_page(original_page)
    
    # Add remaining pages
    for page_num in range(1, len(pdf_reader.pages)):
        pdf_writer.add_page(pdf_reader.pages[page_num])
    
    # Write to output file
    with open(output_pdf, 'wb') as output_file:
        pdf_writer.write(output_file)
    
    print(f"✅ Modified PDF created: {output_pdf}")
    print(f"   Name changed to: Harshit Sharma")
    print(f"   Roll No changed to: 102216014")
    print(f"   Subgroup changed to: 4Q24")

if __name__ == "__main__":
    input_file = "Ansh_Garg_102203808_LabAssign2_CompilerConstruction[1].pdf"
    output_file = "Harshit_Sharma_102216014_LabAssign2_CompilerConstruction.pdf"
    
    try:
        modify_pdf(input_file, output_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nNote: PDF coordinates may need adjustment. The script creates")
        print("white rectangles to cover old text and adds new text on top.")
