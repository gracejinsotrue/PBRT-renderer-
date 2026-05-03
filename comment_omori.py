import re

with open(r'C:\Users\gjin3\Desktop\nori-26sp\scenes\final_scene_real\scene.xml', 'r', encoding='utf-8') as f:
    content = f.read()

def replace_mesh(match):
    val = match.group(0)
    if 'meshes/omori/' in val or 'meshes/tank3.obj' in val:
        return '<!--\n' + val + '\n-->'
    return val

new_content = re.sub(r'<mesh type="obj">.*?</mesh>', replace_mesh, content, flags=re.DOTALL)

with open(r'C:\Users\gjin3\Desktop\nori-26sp\scenes\final_scene_real\scene.xml', 'w', encoding='utf-8') as f:
    f.write(new_content)

commented = new_content.count('<!--\n<mesh type="obj">')
print(f'Commented out {commented} mesh blocks')
