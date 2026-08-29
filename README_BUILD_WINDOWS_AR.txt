تحويل Color Shadow Box Studio إلى برنامج Windows عادي
=====================================================

النتيجة النهائية تمر بمرحلتين:

1) إنشاء البرنامج التنفيذي:
   شغّل BUILD_WINDOWS_EXE.bat

   بعد النجاح ستجد البرنامج هنا:
   dist\ColorShadowBoxStudio\ColorShadowBoxStudio.exe

   هذا المجلد كامل ويجب عدم نقل ملف EXE وحده منه.

2) إنشاء ملف تثبيت واحد Setup.exe:
   ثبّت Inno Setup 6 على Windows، ثم شغّل:
   BUILD_INSTALLER.bat

   بعد النجاح ستجد ملف التثبيت هنا:
   installer_output\ColorShadowBoxStudio_Setup_v1.6.2.exe

ملاحظات مهمة:
- لا تحتاج أجهزة المستخدمين إلى تثبيت Python.
- تم تعديل تشغيل عمليات التحليل والتصدير كي تعمل داخل النسخة المجمعة.
- البناء يجب أن يتم على Windows.
- التطبيق غير موقّع رقميًا حاليًا؛ لذلك قد يعرض Windows رسالة Unknown Publisher.
- لا تحذف ملفات مجلد dist\ColorShadowBoxStudio؛ كلها جزء من البرنامج.
