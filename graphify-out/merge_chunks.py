import json
from pathlib import Path

all_nodes, all_edges, all_hyperedges = [], [], []

ast = json.loads(Path('graphify-out/.graphify_ast.json').read_text())
all_nodes.extend(ast.get('nodes', []))
all_edges.extend(ast.get('edges', []))

cached_path = Path('graphify-out/.graphify_cached.json')
if cached_path.exists():
    cached = json.loads(cached_path.read_text())
    all_nodes.extend(cached.get('nodes', []))
    all_edges.extend(cached.get('edges', []))
    all_hyperedges.extend(cached.get('hyperedges', []))

base = r'c:\Users\felix\AppData\Roaming\Code\User\workspaceStorage\7e5027b5938a1f2a5e8c845e7ba93e6b\GitHub.copilot-chat\chat-session-resources\0a168580-8393-438a-8495-4b93ad091659'
chunk_files = [
    base + r'\toolu_bdrk_01RGnpRV8n3NM9XEHspbRynK__vscode-1777358511447\content.txt',
    base + r'\toolu_bdrk_01YaHBZwWnjW8FJ2xskZVHsa__vscode-1777358511448\content.txt',
    base + r'\toolu_bdrk_01TeVZHTR9fuxN79fFhCqKzj__vscode-1777358511449\content.txt',
    base + r'\toolu_bdrk_01DuAMqjsyvJjwCUq8rUWvmK__vscode-1777358511450\content.txt',
    base + r'\toolu_bdrk_01LDRPCbY1gsUngc8e6GCA7s__vscode-1777358511451\content.txt',
]

for p in chunk_files:
    raw = Path(p).read_text(encoding='utf-8').strip()
    if raw.startswith('```json'):
        raw = raw[7:]
    if raw.endswith('```'):
        raw = raw[:-3]
    raw = raw.strip()
    chunk = json.loads(raw)
    all_nodes.extend(chunk.get('nodes', []))
    all_edges.extend(chunk.get('edges', []))
    all_hyperedges.extend(chunk.get('hyperedges', []))

merged = {'nodes': all_nodes, 'edges': all_edges, 'hyperedges': all_hyperedges, 'input_tokens': 0, 'output_tokens': 0}
Path('graphify-out/.graphify_extract.json').write_text(json.dumps(merged, indent=2))
print('Merged: ' + str(len(all_nodes)) + ' nodes, ' + str(len(all_edges)) + ' edges, ' + str(len(all_hyperedges)) + ' hyperedges')
