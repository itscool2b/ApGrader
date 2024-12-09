from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from PyPDF2 import PdfReader
from .AI import evaluate_essay
from asgiref.sync import sync_to_async

import logging
logger = logging.getLogger(__name__)

@csrf_exempt
def process(request):
    try:
        if request.method == "POST":
            if 'file' not in request.FILES:
                return JsonResponse({'error': 'No file provided'}, status=400)

            pdf_file = request.FILES['file']

            # Ensure the file is a readable PDF
            reader = PdfReader(pdf_file)
            student_essay = "".join([page.extract_text() for page in reader.pages])

            # Process the essay
            response = evaluate_essay(student_essay)
            return JsonResponse({'response': response}, status=200)

        return JsonResponse({'error': 'Invalid request method'}, status=405)
    except Exception as e:
        logger.error(f"Error in process endpoint: {e}")
        return JsonResponse({'error': 'Internal Server Error', 'details': str(e)}, status=500)


