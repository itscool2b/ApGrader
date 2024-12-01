from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from PyPDF2 import PdfReader
from .AI import evaluate_essay
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import sync_to_async

@csrf_exempt
async def process(request):
    if request.method == "POST":
        if 'file' not in request.FILES:
            return JsonResponse({'error':'no file provided bitch'}, status=400)
        
        pdf_file = request.FILES['file']

        try:
            reader = await sync_to_async(PdfReader)(pdf_file)
            student_essay = await sync_to_async(lambda:"".join([page.extract_text() for page in reader.pages]))()
        except Exception as e:
            return JsonResponse({'error': 'Failed to read PDF', 'details': str(e)}, status=500)
        
        try:
            response = await sync_to_async(evaluate_essay)(student_essay)
            
        except Exception as e:
            return JsonResponse({'error': 'AI processing failed', 'details': str(e)}, status=500)

        return JsonResponse({'response': response}, status=200)

    return JsonResponse({'error': 'Invalid request method'}, status=405)