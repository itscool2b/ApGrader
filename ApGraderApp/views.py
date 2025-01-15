from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PyPDF2 import PdfReader
import base64
from asgiref.sync import sync_to_async
import logging
from io import BytesIO
import json
from .ApushLEQ import evaluate
from .ApEuroLEQ import euro_leq_bulk
from .ApushSAQ import evaluate1
from .ApushDBQ import evaluate2
import io
import asyncio
logger = logging.getLogger(__name__)
from PIL import Image
from .ApEuroLEQ import evaluateeuroleq
from .ApEuroSAQ import evaluateeurosaq
from .ApEuroDBQ import evaluateeurodbq
from .ApEuroSAQ import euro_saq_bulk_grading
import zipfile
from django.http import HttpResponse
from .ApEuroDBQ import evaluateeurodbqbulk
from reportlab.pdfgen import canvas








from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

def create_pdf(prompt, response_text):
    pdf_buffer = io.BytesIO()

    doc = BaseDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=72,
        rightMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="content_frame",
        showBoundary=0,
    )

    template = PageTemplate(id="template", frames=[frame])
    doc.addPageTemplates([template])

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="Title",
        fontName="Helvetica-Bold",
        fontSize=28,
        alignment=1,
        spaceAfter=30,
        textColor=colors.HexColor("#1A1A1A"),
        underlineWidth=1.5,
        underlineColor=colors.HexColor("#D4AF37"),
    )

    subtitle_style = ParagraphStyle(
        name="Subtitle",
        fontName="Helvetica-Oblique",
        fontSize=14,
        alignment=1,
        spaceAfter=20,
        textColor=colors.HexColor("#555555"),
    )

    heading_style = ParagraphStyle(
        name="Heading",
        fontName="Helvetica-Bold",
        fontSize=16,
        alignment=0,
        spaceAfter=14,
        textColor=colors.HexColor("#1A1A1A"),
    )

    body_style = ParagraphStyle(
        name="Body",
        fontName="Helvetica",
        fontSize=12,
        alignment=4,
        leading=18,
        spaceAfter=12,
    )

    footer_style = ParagraphStyle(
        name="Footer",
        fontName="Helvetica-Oblique",
        fontSize=10,
        alignment=1,
        textColor=colors.HexColor("#555555"),
    )

    strings_to_bold = [
        "Thesis score", "Contextualization score", "Evidence score",
        "Complex understanding score", "Total summed up score",
        "Thesis feedback", "Contextualization feedback",
        "Evidence feedback", "Complex understanding feedback",
        "Fact-checking feedback", "General Accuracy", 'Feedback',
        'Contextualization feedback', 'Evidence feedback',
        'Evidence beyond feedback', 'Complex understanding feedback',
        'Overall feedback',
    ]

    content = []

    banner = Table(
        [[Paragraph("GRADING REPORT", title_style)]],
        colWidths=[doc.width],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F4F4")),
        ("BOX", (0, 0), (-1, -1), 2, colors.HexColor("#D4AF37")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 20),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
    ]))

    content.append(banner)
    content.append(Spacer(1, 10))
    content.append(Paragraph("AP Essay Evaluation", subtitle_style))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"<b>Prompt:</b> {prompt}", heading_style))
    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Response:</b>", heading_style))
    content.append(Spacer(1, 12))

    for line in response_text.split("\n"):
        for target in strings_to_bold:
            line = line.replace(target, f"<b>{target}</b>")
        content.append(Paragraph(line, body_style))

    content.append(Spacer(1, 24))

    summary_table_data = [
        ["Overall Performance", ""],
        ["Strengths", ""],
        ["Areas for Improvement", ""]
    ]

    summary_table = Table(summary_table_data, colWidths=[2.5 * inch, 4.5 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D4AF37")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D4AF37")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F4F4F4")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    content.append(summary_table)
    content.append(Spacer(1, 30))

    content.append(Paragraph("Generated by the Advanced Grading System", footer_style))

    doc.build(content)
    pdf_buffer.seek(0)
    return pdf_buffer





@csrf_exempt
async def ApushLEQ(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        
        if 'file' in request.FILES:
            pdf_file = request.FILES['file']
            try:
                pdf_stream = BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = "".join([page.extract_text() for page in reader.pages])
                if not essay_text.strip():
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
            except Exception as e:
                logger.error(f"Error processing PDF: {e}")
                return JsonResponse({'error': 'Failed to process the PDF file'}, status=500)
        
        else:
            essay_text = request.POST.get("essay_text", "").strip()
            if not essay_text:
                return JsonResponse({'error': 'Either "essay_file" or "essay_text" is required'}, status=400)
        
        try:
            response = await sync_to_async(evaluate,thread_sensitive=True)(prompt, essay_text)
            logging.info(f"Evaluation successful: {response}")
        except ValueError as e:
            logger.error(f"Evaluation failed: {e}")
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        
        return JsonResponse({
            "response": {
                "output": response
            }
        }, status=200)

    except Exception as e:
        logger.error(f"Error in process endpoint: {e}")
        return JsonResponse({
            "error": "Internal Server Error",
            "details": str(e)
        }, status=500)



@csrf_exempt
async def saq_view(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        
        questions = request.POST.get("questions", "").strip()
        if not questions:
            return JsonResponse({'error': 'Missing "questions" in request'}, status=400)

        
        essay_text = None
        if 'essay_file' in request.FILES:
            
            pdf_file = request.FILES['essay_file']
            try:
                pdf_stream = io.BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = ""
                for page in reader.pages:
                    extracted = page.extract_text() or ""
                    essay_text += extracted
                if not essay_text.strip():
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
            except Exception:
                return JsonResponse({'error': 'Failed to process PDF file'}, status=500)
        else:
            
            essay_text = request.POST.get("essay_text", "").strip()
            if not essay_text:
                return JsonResponse({
                    'error': 'Either "essay_file" or "essay_text" is required'
                }, status=400)

        
        image_data = None
        if 'image' in request.FILES:
            image = request.FILES['image']
            supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
            if image.content_type not in supported_mime_types:
                return JsonResponse({'error': 'Unsupported image type.'}, status=400)

            try:
                image_data = base64.b64encode(image.read()).decode('utf-8')
            except Exception:
                return JsonResponse({'error': 'Failed to process image file.'}, status=500)

        
        try:
            
            response = await sync_to_async(evaluate1,thread_sensitive=True)(questions, essay_text, image_data)
        except Exception as e:
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        return JsonResponse({"response": {"output": response}}, status=200)

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
    

    
from asgiref.sync import sync_to_async

@csrf_exempt
async def bulk_grading_leq(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        prompt = request.POST.get('prompt', '').strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)
        files = request.FILES.getlist('images')
        if not files:
            return JsonResponse({'error': 'No files provided'}, status=400)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                image = file
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': 'Unsupported image type.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                except Exception:
                    return JsonResponse({'error': 'Failed to process image file.'}, status=500)
                try:
                    
                    response_text = await sync_to_async(euro_leq_bulk,thread_sensitive=True)(prompt, image_data)
                    pdf_buffer = create_pdf(prompt, response_text)

                    
                    zip_file.writestr(f"{file.name}_response.pdf", pdf_buffer.read())

                except Exception as e:
                    return JsonResponse({'error': 'Failed to process file', 'details': str(e)}, status=500)

       
        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
                

@csrf_exempt
async def euro_saq_bulk(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        questions = request.POST.get('questions', '').strip()
        if not questions:
            return JsonResponse({'error': 'Missing "questions" in request'}, status=400)
        files = request.FILES.getlist('images')
        if not files:
            return JsonResponse({'error': 'No files provided'}, status=400)

        stim_data = request.FILES.get('stimulus', None)
        if stim_data:

            stim_data = base64.b64encode(stim_data.read()).decode('utf-8')
        else:
            stim_data = None
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                image = file
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': 'Unsupported image type.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                except Exception:
                    return JsonResponse({'error': 'Failed to process image file.'}, status=500)
                try:
                    response_text = await sync_to_async(euro_saq_bulk_grading,thread_sensitive=True)(questions, image_data, stim_data)
                    pdf_buffer = create_pdf(questions, response_text)

                    
                    zip_file.writestr(f"{file.name}_response.pdf", pdf_buffer.read())
                except Exception as e:
                    return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response
    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
async def euro_dbq_bulk(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        images = []
        for key in request.FILES:
            if key.startswith('image_'):
                image = request.FILES[key]
                supported_mime_types = ["image/jpeg", "image/png", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': f'Unsupported image type for {key}.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                    images.append(image_data)
                except Exception as e:
                    return JsonResponse({'error': f'Failed to process {key}: {str(e)}'}, status=500)

        essays = request.FILES.getlist('essays')
        if not essays:
            return JsonResponse({'error': 'No essays provided.'}, status=400)

        
        allowed_image_mime_types = ["image/jpeg", "image/png", "image/webp"]

        for essay in essays:
            if essay.content_type not in allowed_image_mime_types:
                return JsonResponse({
                    'error': f'Unsupported file type for essay "{essay.name}". '
                             f'Allowed types: {allowed_image_mime_types}'
                }, status=400)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for essay in essays:
                try:
                    essay_data = base64.b64encode(essay.read()).decode('utf-8')
                    if not essay_data:
                        return JsonResponse({'error': f'Empty or unreadable essay: {essay.name}'}, status=400)

                    
                    response_text = await sync_to_async(evaluateeurodbqbulk,thread_sensitive=True)(prompt, essay_data, images)
                    pdf_buffer = create_pdf(prompt, response_text)

                    
                    zip_file.writestr(f"{essay.name}_response.pdf", pdf_buffer.read())
                except Exception as e:
                    return JsonResponse({
                        'error': f'Evaluation failed for {essay.name}',
                        'details': str(e)
                    }, status=500)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
from .ApushDBQ import evaluate22
@csrf_exempt

async def apushdbqbulk(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        images = []
        for key in request.FILES:
            if key.startswith('image_'):
                image = request.FILES[key]
                supported_mime_types = ["image/jpeg", "image/png", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': f'Unsupported image type for {key}.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                    images.append(image_data)
                except Exception as e:
                    return JsonResponse({'error': f'Failed to process {key}: {str(e)}'}, status=500)

        essays = request.FILES.getlist('essays')
        if not essays:
            return JsonResponse({'error': 'No essays provided.'}, status=400)

        
        allowed_image_mime_types = ["image/jpeg", "image/png", "image/webp"]

        for essay in essays:
            if essay.content_type not in allowed_image_mime_types:
                return JsonResponse({
                    'error': f'Unsupported file type for essay "{essay.name}". '
                             f'Allowed types: {allowed_image_mime_types}'
                }, status=400)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for essay in essays:
                try:
                    essay_data = base64.b64encode(essay.read()).decode('utf-8')
                    if not essay_data:
                        return JsonResponse({'error': f'Empty or unreadable essay: {essay.name}'}, status=400)

                   
                    response_text = await sync_to_async(evaluate22,thread_sensitive=True)(prompt, essay_data, images)
                    pdf_buffer = create_pdf(prompt, response_text)

                    
                    zip_file.writestr(f"{essay.name}_response.pdf", pdf_buffer.read())
                except Exception as e:
                    return JsonResponse({
                        'error': f'Evaluation failed for {essay.name}',
                        'details': str(e)
                    }, status=500)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)

@csrf_exempt
async def ApEuroLEQ(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        
        if 'file' in request.FILES:
            pdf_file = request.FILES['file']
            try:
                pdf_stream = BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = "".join([page.extract_text() for page in reader.pages])
                if not essay_text.strip():
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
            except Exception as e:
                logger.error(f"Error processing PDF: {e}")
                return JsonResponse({'error': 'Failed to process the PDF file'}, status=500)
        
        else:
            essay_text = request.POST.get("essay_text", "").strip()
            if not essay_text:
                return JsonResponse({'error': 'Either "essay_file" or "essay_text" is required'}, status=400)
        
        try:
            response = await sync_to_async(evaluateeuroleq,thread_sensitive=True)(prompt, essay_text)
            logging.info(f"Evaluation successful: {response}")
        except ValueError as e:
            logger.error(f"Evaluation failed: {e}")
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        
        return JsonResponse({
            "response": {
                "output": response
            }
        }, status=200)

    except Exception as e:
        logger.error(f"Error in process endpoint: {e}")
        return JsonResponse({
            "error": "Internal Server Error",
            "details": str(e)
        }, status=500)

from .ApushLEQ import evaluate69    





@csrf_exempt
async def apushleqbulk(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        prompt = request.POST.get('prompt', '').strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)
        files = request.FILES.getlist('images')
        if not files:
            return JsonResponse({'error': 'No files provided'}, status=400)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                image = file
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': 'Unsupported image type.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                except Exception:
                    return JsonResponse({'error': 'Failed to process image file.'}, status=500)
                try:
                    
                    response_text = await sync_to_async(evaluate69,thread_sensitive=True)(prompt, image_data)
                    pdf_buffer = create_pdf(prompt, response_text)

                    
                    zip_file.writestr(f"{file.name}_response.pdf", pdf_buffer.read())
                except Exception as e:
                    return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response
    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
    

from .ApushSAQ import evaluate11

@csrf_exempt
async def apushsaqbulk(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    try:
        questions = request.POST.get('questions', '').strip()
        if not questions:
            return JsonResponse({'error': 'Missing "questions" in request'}, status=400)
        files = request.FILES.getlist('images')
        if not files:
            return JsonResponse({'error': 'No files provided'}, status=400)

        stim_data = request.FILES.get('stimulus', None)
        if stim_data:

            stim_data = base64.b64encode(stim_data.read()).decode('utf-8')
        else:
            stim_data = None
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                image = file
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': 'Unsupported image type.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                except Exception:
                    return JsonResponse({'error': 'Failed to process image file.'}, status=500)
                try:
                    response_text = await sync_to_async(evaluate11,thread_sensitive=True)(questions, image_data, stim_data)
                    pdf_buffer = create_pdf(questions, response_text)

                    
                    zip_file.writestr(f"{file.name}_response.pdf", pdf_buffer.read())
                except Exception as e:
                    return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="responses.zip"'
        return response
    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)


@csrf_exempt
async def eurosaq_view(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        
        questions = request.POST.get("questions", "").strip()
        if not questions:
            return JsonResponse({'error': 'Missing "questions" in request'}, status=400)

        try:
            questions = json.loads(questions)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format for "questions"'}, status=400)

        
        essay_text = None
        if 'essay_file' in request.FILES:
            pdf_file = request.FILES['essay_file']
            try:
                pdf_stream = io.BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                if not essay_text.strip():
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
            except Exception:
                return JsonResponse({'error': 'Failed to process PDF file'}, status=500)
        else:
            essay_text = request.POST.get("essay_text", "").strip()
            if not essay_text:
                return JsonResponse({'error': 'Either "essay_file" or "essay_text" is required'}, status=400)

       
        image_data = None
        if 'image' in request.FILES:
            image = request.FILES['image']
            supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
            if image.content_type not in supported_mime_types:
                return JsonResponse({'error': 'Unsupported image type.'}, status=400)

            try:
                image_data = base64.b64encode(image.read()).decode('utf-8')
            except Exception:
                return JsonResponse({'error': 'Failed to process image file.'}, status=500)

       
        try:
            response = await sync_to_async(evaluateeurosaq,thread_sensitive=True)(questions, essay_text,image_data)
        except Exception as e:
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        
        return JsonResponse({"response": {"output": response}}, status=200)

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
    
@csrf_exempt
async def eurodbq(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        essay_text = ""
        if 'essays' in request.FILES: 
            try:
                pdf_file = request.FILES['essays']
                pdf_stream = io.BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                if not essay_text.strip():
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
            except Exception as e:
                return JsonResponse({'error': f'Failed to process PDF file: {str(e)}'}, status=400)
        else:  
            essay_text = request.POST.get("essays", "").strip()
            if not essay_text:
                return JsonResponse({'error': 'Either a PDF file or essay text is required'}, status=400)

        
        images = []
        for i in range(1, 8):
            image_key = f'image_{i}'
            if image_key in request.FILES:
                image = request.FILES[image_key]
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': f'Unsupported image type for {image_key}.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                    images.append(image_data)
                except Exception as e:
                    return JsonResponse({'error': f'Failed to process {image_key}: {str(e)}'}, status=500)

        
        images = images[:7] + [None] * (7 - len(images))

        
        try:
            response = await sync_to_async(evaluateeurodbq,thread_sensitive=True)(prompt, essay_text, images)
        except Exception as e:
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        return JsonResponse({"response": {"output": response}}, status=200)

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
@csrf_exempt
async def dbq_view(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        essay_text = ""
        if 'essays' in request.FILES: 
            try:
                pdf_file = request.FILES['essays']
                pdf_stream = io.BytesIO(pdf_file.read())
                reader = PdfReader(pdf_stream)
                essay_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                if not essay_text.strip():
                    return JsonResponse({'error': 'Empty or unreadable PDF file'}, status=400)
            except Exception as e:
                return JsonResponse({'error': f'Failed to process PDF file: {str(e)}'}, status=400)
        else:  
            essay_text = request.POST.get("essays", "").strip()
            if not essay_text:
                return JsonResponse({'error': 'Either a PDF file or essay text is required'}, status=400)

        
        images = []
        for i in range(1, 8):
            image_key = f'image_{i}'
            if image_key in request.FILES:
                image = request.FILES[image_key]
                supported_mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
                if image.content_type not in supported_mime_types:
                    return JsonResponse({'error': f'Unsupported image type for {image_key}.'}, status=400)
                try:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                    images.append(image_data)
                except Exception as e:
                    return JsonResponse({'error': f'Failed to process {image_key}: {str(e)}'}, status=500)

        
        images = images[:7] + [None] * (7 - len(images))

        
        try:
            response = await sync_to_async(evaluate2,thread_sensitive=True)(prompt, essay_text, images)
        except Exception as e:
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        return JsonResponse({"response": {"output": response}}, status=200)

    except Exception as e:
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)

@csrf_exempt
async def textbulk(request):
    print("[DEBUG] textbulk endpoint called")

    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method. Only POST is allowed.'}, status=405)

    content_type = request.META.get('CONTENT_TYPE', '').lower()
    print(f"[DEBUG] content_type='{content_type}'")

    data = {}
    submission_type = None

    if 'application/json' in content_type:
        try:
            data = json.loads(request.body)
            submission_type = data.get('submission_type', '').strip()
            print(f"[DEBUG] JSON parsed successfully, submission_type='{submission_type}'")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[DEBUG] JSON decode error => {str(e)}")
            return JsonResponse({'error': f'Invalid JSON data: {str(e)}'}, status=400)
    elif 'multipart/form-data' in content_type:
        submission_type = request.POST.get('submission_type', '').strip()
        print(f"[DEBUG] multipart form => submission_type='{submission_type}'")
    else:
        print(f"[DEBUG] Unsupported content type => {content_type}")
        return JsonResponse({'error': f'Unsupported content type: {content_type}'}, status=400)

    print(f"[DEBUG] Final submission_type='{submission_type}'")

    if submission_type == 'apushleq':
        if 'application/json' in content_type:
            essays = data.get('essays', [])
            prompt = data.get('prompt', '').strip()
        else:
            prompt = request.POST.get('prompt', '').strip()
            essays = json.loads(request.POST.get('essays', '[]'))
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for essay in essays:
                response_text = await sync_to_async(evaluate,thread_sensitive=True)(prompt, essay)
                pdf_buffer = create_pdf(prompt, response_text)
                zip_file.writestr(f"{essay.get('name', 'Untitled')}_response.pdf", pdf_buffer.read())
        zip_buffer.seek(0)
        return HttpResponse(zip_buffer, content_type='application/zip')

    elif submission_type == 'apeuroleq':
        if 'application/json' in content_type:
            essays = data.get('essays', [])
            prompt = data.get('prompt', '').strip()
        else:
            prompt = request.POST.get('prompt', '').strip()
            essays = json.loads(request.POST.get('essays', '[]'))
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for essay in essays:
                response_text = await sync_to_async(evaluateeuroleq,thread_sensitive=True)(prompt, essay)
                pdf_buffer = create_pdf(prompt, response_text)
                zip_file.writestr(f"{essay.get('name', 'Untitled')}_response.pdf", pdf_buffer.read())
        zip_buffer.seek(0)
        return HttpResponse(zip_buffer, content_type='application/zip')

    elif submission_type == 'apushsaq':
        prompt = ""
        essays = []
        stim_data = None
        if 'application/json' in content_type:
            prompt = data.get('questions', '').strip()
            essays = data.get('essays', [])
        else:
            prompt = request.POST.get('questions', '').strip()
            essays = json.loads(request.POST.get('essays', '[]'))
            uploaded_image = request.FILES.get('image')
            if uploaded_image:
                stim_data = base64.b64encode(uploaded_image.read()).decode('utf-8')
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for essay in essays:
                response_text = await sync_to_async(evaluate1,thread_sensitive=True)(prompt, essay, stim_data)
                pdf_buffer = create_pdf(prompt, response_text)
                zip_file.writestr(f"{essay.get('name', 'Untitled')}_response.pdf", pdf_buffer.read())
        zip_buffer.seek(0)
        return HttpResponse(zip_buffer, content_type='application/zip')

    elif submission_type == 'apeurosaq':
        prompt = ""
        essays = []
        stim_data = None
        if 'application/json' in content_type:
            prompt = data.get('questions', '').strip()
            essays = data.get('essays', [])
        else:
            prompt = request.POST.get('questions', '').strip()
            essays = json.loads(request.POST.get('essays', '[]'))
            uploaded_image = request.FILES.get('image')
            if uploaded_image:
                stim_data = base64.b64encode(uploaded_image.read()).decode('utf-8')
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for essay in essays:
                response_text = await sync_to_async(evaluateeurosaq,thread_sensitive=True)(prompt, essay, stim_data)
                pdf_buffer = create_pdf(prompt, response_text)
                zip_file.writestr(f"{essay.get('name', 'Untitled')}_response.pdf", pdf_buffer.read())
        zip_buffer.seek(0)
        return HttpResponse(zip_buffer, content_type='application/zip')

    elif submission_type == 'apushdbq':
        prompt = ""
        essays = []
        images = []
        if 'application/json' in content_type:
            prompt = data.get('prompt', '').strip()
            essays = data.get('essays', [])
        else:
            prompt = request.POST.get('prompt', '').strip()
            essays = json.loads(request.POST.get('essays', '[]'))
            for i in range(1, 8):
                image_key = f'image_{i}'
                if image_key in request.FILES:
                    image_file = request.FILES[image_key]
                    images.append(base64.b64encode(image_file.read()).decode('utf-8'))
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for essay in essays:
                response_text = await sync_to_async(evaluate2,thread_sensitive=True)(prompt, essay, images)
                pdf_buffer = create_pdf(prompt, response_text)
                zip_file.writestr(f"{essay.get('name', 'Untitled')}_response.pdf", pdf_buffer.read())
        zip_buffer.seek(0)
        return HttpResponse(zip_buffer, content_type='application/zip')

    elif submission_type == 'apeurodbq':
        prompt = ""
        essays = []
        images = []
        if 'application/json' in content_type:
            prompt = data.get('prompt', '').strip()
            essays = data.get('essays', [])
        else:
            prompt = request.POST.get('prompt', '').strip()
            essays = json.loads(request.POST.get('essays', '[]'))
            for i in range(1, 8):
                image_key = f'image_{i}'
                if image_key in request.FILES:
                    image_file = request.FILES[image_key]
                    images.append(base64.b64encode(image_file.read()).decode('utf-8'))
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for essay in essays:
                response_text = await sync_to_async(evaluateeurodbq,thread_sensitive=True)(prompt, essay, images)
                pdf_buffer = create_pdf(prompt, response_text)
                zip_file.writestr(f"{essay.get('name', 'Untitled')}_response.pdf", pdf_buffer.read())
        zip_buffer.seek(0)
        return HttpResponse(zip_buffer, content_type='application/zip')

    return JsonResponse({'error': 'Invalid or missing submission_type.'}, status=400)
