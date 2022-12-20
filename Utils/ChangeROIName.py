import pandas as pd
import sys,os
from omero.api import RoiOptions
from omero.rtypes import rstring, rlong, unwrap, rdouble, rint
from omero.gateway import BlitzGateway, Delete2
from omero.cli import cli_login, CLI
import omero
from omero.cmd import DiskUsage2
from omero.cli import CmdControl
import toml
from pathlib import Path

pd.set_option('display.max_rows', None)
def connect(hostname, username, password, **kwargs):
    conn = BlitzGateway(username, password, host=hostname, secure=True, **kwargs)
    conn.connect()
    conn.c.enableKeepAlive(60)
    return conn

config = toml.load(sys.argv[1])
NameToChange = sys.argv[2]
NewName = sys.argv[3]
conn = connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw'])  ## Group not implemented yet

conn.SERVICE_OPTS.setOmeroGroup('-1')
updateService = conn.getUpdateService()

query = """
select roi.id, roi.image.id,shapes.textValue from  Roi as roi
left outer join roi.shapes as shapes  
"""
rois = conn.getQueryService().projection(query, None,{"omero.group": "-1"})
df   = pd.DataFrame([[roi[0].val, roi[1].val, roi[2].val] for roi in rois],columns=["ID","Image","Name"])
df   = df[~df['Name'].str.contains('MF')]
df   = df[~df['Name'].str.contains('TP')]
df   = df[~df['Name'].str.contains('FP')]
df   = df[~df['Name'].str.contains('MAF')]
df   = df[~df['Name'].str.contains('Ambiguous')].reset_index()
df.to_csv("FromChangeROIName.csv")
print(df)
print(set(df['Name']))
df['Name']   = df['Name'].str.split().str.join(' ')

#print(df)

df = df[df['Name']==NameToChange].reset_index()


print(sorted(set(df['Name'])))

roi_service = conn.getRoiService()
roi_options = RoiOptions(groupId=rlong("-1"))

print(df)
for idx, row in df.iterrows():
    results = roi_service.findByRoi(row['ID'],roi_options)
    #results = roi_service.findByImage(row['Image'],roi_options)
    for roi in results.rois:
        for s in roi.copyShapes():
            s.setTextValue(rstring(NewName))

        #roi = updateService.saveAndReturnObject(roi)
            


conn.close()
print('done')
