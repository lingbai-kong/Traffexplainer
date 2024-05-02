# 绘制矩阵            
import matplotlib.pyplot as plt
from matplotlib import cm,colors
from progressbar import ProgressBar,Percentage,Bar,Timer,ETA,FileTransferSpeed
import numpy as np
def print_mx(mx,orientation= 'vertical'):
    mx=np.array(mx)
    mx[mx==0]=1
    mx[mx<0.9]=0.9
    mx[mx>1.1]=1.1
    minium,maxium=min(np.array(mx).flatten()),max(np.array(mx).flatten())
    fig,ax = plt.subplots(figsize=(8,5),dpi=200)
    ax.matshow(mx,cmap='seismic')
    norm = colors.Normalize(vmin=minium, vmax=maxium)
    im = cm.ScalarMappable(norm=norm, cmap='seismic')
    fig.colorbar(im,ax=ax,orientation=orientation) 
    fig.show()

def print_vector(array,orientation= 'vertical'):
    minium,maxium=0.9,1.1
    fig,ax = plt.subplots(figsize=(8,2),dpi=200)
    ax.matshow(array,cmap='viridis')
    norm = colors.Normalize(vmin=minium, vmax=maxium)
    im = cm.ScalarMappable(norm=norm, cmap='viridis')
    ax.set_xticks([i for i in range(12)])
    ax.set_xticklabels([str(-60+5*i) for i in range(12)])
    ax.set_yticks([])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    # ax.set_title('Average Temporal Mask', fontsize=16)
    fig.colorbar(im,ax=ax, orientation=orientation)
    fig.show()
    
def print_scatter(file_name,mx):
    mx=np.array(mx)
    minium,maxium=0.9,1.1
    x,y,c=[],[],[]
    
    for i in range(len(mx)):
        for j in range(len(mx)):
            if mx[i][j]>0 and i!=j:
                v=min(max(mx[i][j],minium),maxium)
                x.append(i)
                y.append(j)
                c.append(v) 
    fig,ax = plt.subplots(figsize=(5,4),dpi=200)
    ax.scatter(x,y,c=c,s=0.1,cmap='seismic')
    
    ax.set_ylabel('Node No.',fontsize=8)
    ax.set_xlabel('Node No.',fontsize=8)
    
    norm = colors.Normalize(vmin=minium, vmax=maxium)
    im = cm.ScalarMappable(norm=norm, cmap='seismic')
    fig.colorbar(im,ax=ax)
    
    ax.tick_params(labelsize=12)
    fig.savefig(file_name,dpi=fig.dpi,bbox_inches='tight')
    fig.show()
def print_array(array,orientation= 'vertical'):
    minium,maxium=min(np.array(array).flatten()),max(np.array(array).flatten())
    fig,ax = plt.subplots(figsize=(8,2),dpi=200)
    ax.matshow(array,cmap='viridis')
    norm = colors.Normalize(vmin=minium, vmax=maxium)
    im = cm.ScalarMappable(norm=norm, cmap='viridis')
    ax.set_xticks([i for i in range(12)])
    ax.set_xticklabels([str(-60+5*i) for i in range(12)])
    ax.set_yticks([])
    ax.xaxis.tick_top()
    ax.set_xlabel('time slot(min)',fontsize=14)
    ax.xaxis.set_label_position('top') 
    # ax.set_title('Average Temporal Mask', fontsize=16)
    fig.colorbar(im,ax=ax, orientation=orientation)
    fig.savefig('../img/array.png',dpi=fig.dpi, bbox_inches='tight')
    fig.show()
    
def draw_matrix(file_name,mask,graph,ref,geo_ids):
    widgets = [
        'Progress: ',
        Percentage(), ' ',
        Bar('#'), ' ',
        Timer(), ' ',
        ETA(), ' ',
        FileTransferSpeed()
    ]

    mask=np.array(mask)
    mask[mask==0]=1
    mask[mask<0.9]=0.9
    mask[mask>1.1]=1.1
    fig,ax = plt.subplots(figsize=(12,5),dpi=200)
    for v,e in graph.items():
        for oute in e[1]:
            ax.plot([float(ref[v][0]),float(ref[oute[1]][0])],
                    [float(ref[v][1]),float(ref[oute[1]][1])],
                    color='black',
                    linewidth=.1)

    bar = ProgressBar(widgets=widgets, maxval=len(mask)*len(mask)).start()
    mask[np.isnan(mask)]=0
    mask[np.isinf(mask)]=0
    maxium,minium=np.max(mask),np.min(mask)
    cmap = cm.get_cmap('seismic')
    for i in range(len(mask)):
        for j in range(len(mask)):
            if mask[i][j]!=1:
                v_i = str(geo_ids[i])
                v_j = str(geo_ids[j])
                ax.plot([float(ref[v_i][0]),float(ref[v_j][0])],
                        [float(ref[v_i][1]),float(ref[v_j][1])],
                        color=cmap(float((mask[i][j]-minium)/(maxium-minium))),
                        linewidth=1)
                bar.update(i*len(mask)+j+1)
    bar.finish()

    ax.tick_params(direction='out',labelsize=20,length=6.5,width=1,top=False,right=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Latitude',fontsize=13)
    ax.set_xlabel('Longitude',fontsize=13)
    text_font = {'size':'17','weight':'bold','color':'black'}
    # ax.text(.03,.93,'(Road Graph)',transform = ax.transAxes,fontdict=text_font,zorder=4)
    # ax.text(.87,-.08,'\nVisualization by Jupyter',transform = ax.transAxes, ha='center', va='center',fontsize = 5,color='black',fontweight='bold')
    norm = colors.Normalize(vmin=minium, vmax=maxium)
    im = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(im,ax=ax)
    fig.savefig(file_name, bbox_inches='tight')
    fig.show()
def draw_matrix_btmlmt(file_name,mask,graph,ref,geo_ids,t=1,h=2):
    widgets = [
        'Progress: ',
        Percentage(), ' ',
        Bar('#'), ' ',
        Timer(), ' ',
        ETA(), ' ',
        FileTransferSpeed()
    ]

    mask=np.array(mask)
    mask[mask>h]=h
    fig,ax = plt.subplots(figsize=(8,5),dpi=200)
    for v,e in graph.items():
        for oute in e[1]:
            ax.plot([float(ref[v][0]),float(ref[oute[1]][0])],
                    [float(ref[v][1]),float(ref[oute[1]][1])],
                    color='black',
                    linewidth=.1)

    bar = ProgressBar(widgets=widgets, maxval=len(mask)*len(mask)).start()
    mask[np.isnan(mask)]=0
    mask[np.isinf(mask)]=0
    maxium,minium=np.max(mask),t
    cmap = cm.get_cmap('viridis')
    for i in range(len(mask)):
        for j in range(len(mask)):
            if mask[i][j]>t:
                v_i = str(geo_ids[i])
                v_j = str(geo_ids[j])
                ax.plot([float(ref[v_i][0]),float(ref[v_j][0])],
                        [float(ref[v_i][1]),float(ref[v_j][1])],
                        color=cmap(float((mask[i][j]-minium)/(maxium-minium))),
                        linewidth=1)
                bar.update(i*len(mask)+j+1)
    bar.finish()

    ax.tick_params(direction='out',labelsize=20,length=6.5,width=1,top=False,right=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Latitude',fontsize=13)
    ax.set_xlabel('Longitude',fontsize=13)
    text_font = {'size':'17','weight':'bold','color':'black'}
    # ax.text(.03,.93,'(Road Graph)',transform = ax.transAxes,fontdict=text_font,zorder=4)
    # ax.text(.87,-.08,'\nVisualization by Jupyter',transform = ax.transAxes, ha='center', va='center',fontsize = 5,color='black',fontweight='bold')
    norm = colors.Normalize(vmin=minium, vmax=maxium)
    im = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(im,ax=ax)
    fig.savefig(file_name, bbox_inches='tight',dpi=fig.dpi)
    fig.show()
def draw_distribute(tensor):
    tensor=tensor.reshape(-1)
    tensor=tensor[tensor>0]
    plt.hist(tensor)
    plt.show()