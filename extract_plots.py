import json, base64, os

nb = json.load(open('Market_Price_Forecasting.ipynb', 'r', encoding='utf-8'))
os.makedirs('slides_img', exist_ok=True)

count = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    for output in cell.get('outputs', []):
        if output.get('output_type') in ('display_data', 'execute_result'):
            data = output.get('data', {})
            if 'image/png' in data:
                img_data = base64.b64decode(data['image/png'])
                fname = f'slides_img/plot_{count:02d}.png'
                with open(fname, 'wb') as f:
                    f.write(img_data)
                # Get context from cell source
                src = ''.join(cell['source'])[:80]
                print(f'{fname} ({len(img_data)} bytes) - {src}')
                count += 1

print(f'\nTotal images extracted: {count}')
