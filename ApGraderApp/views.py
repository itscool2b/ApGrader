from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PyPDF2 import PdfReader
from .AI import evaluate_essay
from asgiref.sync import sync_to_async
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

@csrf_exempt
async def process(request):
    try:
        if request.method == "POST":
            if 'file' not in request.FILES:
                return JsonResponse({'error': 'No file provided'}, status=400)

            pdf_file = request.FILES['file']

            # Convert the uploaded file into a BytesIO stream
            pdf_stream = BytesIO(pdf_file.read())

            # Initialize PdfReader with the BytesIO stream
            reader = PdfReader(pdf_stream)

            # Extract text from each page
            student_essay = "".join([page.extract_text() for page in reader.pages])
            logger.info("Completed PDF text extraction")

            # Process the essay asynchronously
            response = await sync_to_async(evaluate_essay)(student_essay)

            return JsonResponse({'response': response}, status=200)

        return JsonResponse({'error': 'Invalid request method'}, status=405)
    except Exception as e:
        logger.error(f"Error in process endpoint: {e}")
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)
