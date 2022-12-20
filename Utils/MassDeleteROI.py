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


def connect(hostname, username, password, **kwargs):
    conn = BlitzGateway(username, password, host=hostname, secure=True, **kwargs)
    conn.connect()
    conn.SERVICE_OPTS.setOmeroGroup('-1')
    conn.c.enableKeepAlive(60)
    return conn

config = toml.load(sys.argv[1])
df     = pd.read_csv(sys.argv[2])

conn = connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw'])  ## Group not implemented yet                                                                                          
conn.SERVICE_OPTS.setOmeroGroup('-1')


#print(dir(conn.SERVICE_OPTS))

#updateService = conn.getUpdateService()


print("Querying ROI from Server")
df   = pd.DataFrame()

query = """
select roi.id,roi.image.id,shapes.textValue from
Roi as roi
left join roi.shapes as shapes
"""                                                                                                                                                                                                       
rois   = conn.getQueryService().projection(query, None,{"omero.group": "-1"})
df     = pd.DataFrame([[roi[0].val, roi[1].val, roi[2].val] for roi in  rois], columns=["ID","Image","Name"])
print(df)
new_df = df[df['Name']=='tissue'].reset_index()
#new_df = df[df['Name'].isnull()].reset_index()
#new_df2 = new_df[new_df['Image']==201].reset_index()
to_del  = list(new_df['ID'])
print(new_df, to_del)

handle = conn.deleteObjects("Roi", to_del)
cb = omero.callbacks.CmdCallbackI(conn.c, handle)
print("Deleting, please wait.")
while not cb.block(500):
    print(".")
err = isinstance(cb.getResponse(), omero.cmd.ERR)
print("Error?", err)
if err:
    print(cb.getResponse())
cb.close(True)      # close handle too    
print(handle)
print('done')

conn.close()


