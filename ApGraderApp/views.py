from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from PyPDF2 import PdfReader
from .AI import evaluate_essay
from asgiref.sync import sync_to_async

@csrf_exempt
async def process(request):
    if request.method == "POST":
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file provided'}, status=400)
        
        pdf_file = request.FILES['file']

        try:
            # Reading PDF asynchronously
            reader = await sync_to_async(PdfReader)(pdf_file)
            # Extract text asynchronously
            student_essay = await sync_to_async(lambda: "".join([page.extract_text() for page in reader.pages]))()
        except Exception as e:
            return JsonResponse({'error': 'Failed to read PDF', 'details': str(e)}, status=500)
        
        try:
            # Evaluate essay asynchronously
            response = await sync_to_async(evaluate_essay)(student_essay)
            
        except Exception as e:
            return JsonResponse({'error': 'AI processing failed', 'details': str(e)}, status=500)

        return JsonResponse({'response': response}, status=200)

    return JsonResponse({'error': 'Invalid request method'}, status=405)
