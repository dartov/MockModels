from aoa import ModelContext
import png
def train(context: ModelContext, **kwargs):
    s = ['110010010011',
         '101011010100',
         '110010110101',
         '100010010011']
    s = [[int(c) for c in row] for row in s]

    palette=[(0x55,0x55,0x55), (0xff,0x99,0x99)]
    w = png.Writer(len(s[0]), len(s), palette=palette, bitdepth=1)
    f = open(f"{context.artifact_output_path}/model_chart.png", 'wb')
    w.write(f, s)
    f.close()
