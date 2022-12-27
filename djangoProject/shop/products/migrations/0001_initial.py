# Generated by Django 4.1.4 on 2022-12-27 00:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('categories', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ShopProduct',
            fields=[
                ('product_id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.TextField()),
                ('price', models.IntegerField()),
                ('image_url', models.TextField()),
                ('shop_category', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='categories.shopcategory')),
            ],
            options={
                'db_table': 'shop_products',
            },
        ),
    ]
