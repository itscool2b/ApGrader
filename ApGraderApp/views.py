from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PyPDF2 import PdfReader

from asgiref.sync import sync_to_async
import logging
from io import BytesIO
import json
from .ApushLEQ import evaluate  # Updated import
from .ApushSAQ import evaluate1
logger = logging.getLogger(__name__)

@csrf_exempt
async def ApushLEQ(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        # Fetch prompt from request
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({'error': 'Missing "prompt" in request'}, status=400)

        # Fetch and process PDF file
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
            return JsonResponse({'error': 'PDF file is required'}, status=400)

        # Evaluate the essay using the prompt
        try:
            response = await sync_to_async(evaluate)(prompt, essay_text)
            logging.info(f"Evaluation successful: {response}")
        except ValueError as e:
            logger.error(f"Evaluation failed: {e}")
            return JsonResponse({'error': 'Evaluation failed', 'details': str(e)}, status=500)

        # Return response in the specified JSON format
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
    """
    View to handle SAQ submissions with JSON questions, PDF essay, and optional image.
    """
    if request.method == "POST":
        
        try:
            questions_json = request.POST.get("questions")
            if not questions_json:
                raise ValueError("Missing 'questions' in the request.")
            questions = json.loads(questions_json)  
        except Exception as e:
            return JsonResponse({"error": f"Invalid or missing JSON questions: {str(e)}"}, status=400)

        
        essay_file = request.FILES.get("essay_file")
        if not essay_file:
            return JsonResponse({"error": "The 'essay_file' (PDF) is required."}, status=400)

        try:
            
            pdf_reader = PdfReader(io.BytesIO(essay_file.read()))
            essay_text = " ".join([page.extract_text() for page in pdf_reader.pages]).strip()
            if not essay_text:
                raise ValueError("The provided PDF is empty or cannot be processed.")
        except Exception as e:
            return JsonResponse({"error": f"Error reading PDF: {str(e)}"}, status=400)

        
        image = request.FILES.get("image")
        image_data = image.read() if image else None

        try:
            

            response = await sync_to_async(evaluate1)(questions,essay_text,image_data)
            return JsonResponse({"message": "Evaluation completed successfully.", "result": response})
        except Exception as e:
            return JsonResponse({"error": f"Error during evaluation: {str(e)}"}, status=500)

    return JsonResponse({"error": "Only POST method is allowed."}, status=405)


