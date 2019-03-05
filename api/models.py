from django.db import models
class Note(models.Model):
  name = models.CharField(max_length=200)
def _str_(self):
   return '%s' % (self.name)
# Create your models here.
