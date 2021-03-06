
# https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Figure.html
# https://plotly.com/python/shapes/
# fig = go.Figure()

# # Set axes properties
# fig.update_xaxes(range=[0, 7], showgrid=False)
# fig.update_yaxes(range=[0, 3.5])

# # Add shapes
# fig.add_shape(
#                 #type="rect",
#                 x0=1, y0=1, x1=3, y1=2,
#                 line=dict(color="RoyalBlue"),
#                 fillcolor="RoyalBlue",
#                 opacity=1
                
# )

# fig.add_annotation(
#                 showarrow=False,
#                 text="this is a claim slslslslfsf",
#                 x = 2,
#                 y = 1.5,
    
#             )
# fig.update_shapes(dict(xref='x', yref='y'))
# fig.show()


#def tree_graph(sample:Union[ModelInput, ModelOutput])




#basics
from copy import deepcopy
import numpy as np
import random
import math

#plotly
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib as mpl


class TextNode:

    def __init__(self,  ID="ROOT", 
                        text="", 
                        label=None,
                        label_color="grey",
                        link=None, 
                        link_label=None,
                        link_label_color="grey",
                        ):
        self.id = ID
        self.label = label
        self.link = link
        self.link_label = link_label
        self.label_color = label_color
        self.link_label_color = link_label_color
        self.text = text
        self.children = []
        self.__measured = False
        self.level_width = 0
        self._fig = None

    @property
    def max_width(self):
        if self.__measured:
            return self._max_width
        else:
            self.__measure()
            return self._max_width


    @property
    def max_depth(self):
        if self.__measured:
            return self._max_depth
        else:
            self.__measure()
            return self._max_depth


    def __measure(self, start=0, widths={}, linklabelcolors:set=set(), labelcolors:set=set()):
        self.depth = start
        self.widths = widths
        max_depth = start
    
        if start not in self.widths:
            self.widths[self.depth] = 0
        
        self.level_pos = self.widths[self.depth]
        self.widths[self.depth] += 1

        linklabelcolors.add((self.link_label, self.link_label_color))
        labelcolors.add((self.label, self.label_color))

        level_width = len(self.children)
        for i,child in enumerate(self.children):

            child_depth = child.__measure(start=start+1, widths=self.widths, linklabelcolors=linklabelcolors, labelcolors=labelcolors)

            if child_depth > max_depth:
                max_depth = child_depth

        self.__measured = True
        self._max_depth = max_depth
        self._max_width = max(list(self.widths.values()))
        self.linklabelcolors = linklabelcolors
        self.labelcolors = labelcolors
        return max_depth


    def get_xs(self, width):
        num = 3 if width == 0 else (width*2) +1
        xs = [v for i,v in enumerate(np.linspace(0, 100, num=num)[1:-1:], num) if i == 0 or i % 2 != 0]
        return xs

    
    def show(self):

        if not self.__measured:
            self.__measure()

        if self._fig is not None:
            self._fig.show()
        else:
            self._fig = self.__make_fig(self)
            self._fig.update_layout(
                                    autosize=False,
                                    width=3000,
                                    height=800,
                                    margin=dict(
                                        # l=50,
                                        # r=50,
                                        # b=50,
                                        # t=50,
                                        pad=4
                                        ),
                                    paper_bgcolor="white",
                                    plot_bgcolor="white",
                                    showlegend=False,
                                    )
            self._fig.update_xaxes(showgrid=False, showticklabels=False)
            self._fig.update_yaxes(showgrid=False, showticklabels=False)
            self.__add_costum_legend(self._fig)
            self._fig.show()


    def __add_costum_legend(self, fig):
        
        i = 0 
        nr_link_lables = 0
        for l, c in self.linklabelcolors:

            if l is None or l == "None":
                continue

            fig.add_trace(dict(
                                x=[i],
                                y=[-1],
                                mode="markers+text",
                                marker=dict(
                                            color=c,
                                            size=20,
                                            symbol='square',
                                                ),
                                text=[l],
                                textposition="top center",
                                    )
                            )   
            nr_link_lables += 1
            i += 3


        fig.add_trace(dict(
                            x=[(i-3)/nr_link_lables],
                            y=[-0.5],
                            mode="text",
                            text=["Link Labels"],
                            textposition="bottom center",
                                )
                        )   
        
        fig.add_trace(dict(
                            x=[i-1.5]*2,
                            y=[-0.4, -1.2],
                            mode="lines",
                            line=dict(color="black")
                            )
                        )   

        j = i
        nr_lables = 0
        for l, c in self.labelcolors:

            if l is None or l == "None":
                continue
    
            fig.add_trace(dict(
                                x=[j],
                                y=[-1],
                                mode="markers+text",
                                marker=dict(
                                            color=c,
                                            size=20,
                                            symbol='square',
                                                ),
                                text=[l],
                                textposition="top center",
                                    )
                            )  
            nr_lables += 1   
            j += 3


        fig.add_trace(dict(
                            x=[(j-3) - nr_lables],
                            y=[-0.5],
                            mode="text",
                            text=["Labels"],
                            textposition="bottom center",
                                )
                        )   


        fig.add_trace(dict(
                    x=[-1, j-1,  j-1,  j-1, j-1, -1, -1, -1],
                    y=[-0.4, -0.4, -0.4, -1.2, -1.2, -1.2, -1.2, -0.4],
                    mode="lines",
                    line=dict(color="black")
                )
                )   


    def __get_text_box(self):

        text_size = len(self.text)

        line_length = 60
        line_height = 0.2

        nr_lines = math.ceil(text_size / line_length)
        height = line_height*nr_lines  
        max_height = height/2
        current_line_pos = max_height
        start = 0
        line_pos = []

        self.text += " "
        start = 0
        for i  in range(0, nr_lines+1):

            if start+line_length > text_size:
                line = self.text[start:]
                line_pos.append((line, current_line_pos))
                break
            else:
                look_from = start+line_length-5
                closes_space = self.text.find(" ", look_from)
                line = self.text[start:closes_space+1]
                start = closes_space

                line_pos.append((line, current_line_pos))
            current_line_pos -= line_height

        
        return line_pos, height+(line_height*2), (line_length/3) -8, line_height


    def __make_fig(self, fig=None, grid=None, p_xy=None):


        xs = self.get_xs(self.widths[self.depth])
        x = xs[self.level_pos]
        y = (self.depth - 1 ) * 2

        if  self.id == "ROOT":
            fig = go.Figure()
        else:
     
            line_pos, box_hight, box_length, line_height = self.__get_text_box()

            if self.id != "ROOT" and self.link != "ROOT":
                fig.add_trace(self.line(
                                        x=[p_xy[0], x],
                                        y=[p_xy[1], y]
                                        )
                                )


            x0 = x - (box_length /2)
            y0 = y - (box_hight/2) + line_height
            x1 = x + (box_length /2)
            y1 = y + (box_hight/2)
            fig.add_shape(self.box(
                                    x0=x0, 
                                    y0=y0, 
                                    x1=x1,
                                    y1=y1, 
                                    ))

            for line, pos in line_pos:
                fig.add_annotation(self.annotation(
                                                    x=x,
                                                    y=y+pos,
                                                    text=line
                                                    ))


        for child in self.children:
            child.__make_fig(fig=fig, grid=grid, p_xy=(x,y))

        return fig


    def line(self, x:list, y:list):
        return dict(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict( #go.scatter.Line(
                                color=self.link_label_color,
                                ),
                    name=self.link_label,
                    legendgroup=self.link_label,  # this can be any string, not just "group"
                    )
            

    def box(self, x0:float, y0:float, x1:float, y1:float):
        return dict(
                    #type="rect",
                    x0=x0, 
                    y0=y0, 
                    x1=x1, 
                    y1=y1,
                    line=dict(
                                color=self.label_color,
                                ),
                    fillcolor=self.label_color,
                    opacity=1,
                    name=self.label
                    )


    def annotation(self, x:float, y:float, text:str):
        return dict(
                        showarrow=False,
                        text=text,
                        x = x,
                        y = y,
                        font=dict(
                                    #family="Courier New, monospace",
                                    size=10,
                                    ),
                    )



def create_tree(tree:TextNode, nodes:list):
    for i, node in enumerate(nodes):
        if node.link == tree.id:
            filtered = nodes.copy()
            filtered.pop(i)
            tree.children.append(create_tree( 
                                            tree=node, 
                                            nodes=filtered
                                            ))
    return tree



def arrays_to_tree(
                    span_lengths:list, 
                    span_token_lengths:list,
                    none_span_mask:list,
                    links:list,
                    labels:list,
                    tokens:list,
                    label_colors:dict,
                    link_labels=None,
                    ):
    nodes = []
    start = 0
    j = 0

    for i in range(span_lengths):
        
        length = span_token_lengths[i]

        #print(none_span_mask[i])
        if none_span_mask[i]:

            link = links[start:start+length][0]
            label = labels[start:start+length][0]

            text  = " ".join(tokens[start:start+length])

            if link_labels is not None:
                link_label= link_labels[start:start+length][0]

            if j == link:
                link = "ROOT"
            
            nodes.append(TextNode(
                                    ID=j,
                                    link=link, 
                                    label=label,
                                    label_color=label_colors[label],
                                    link_label=link_label,
                                    link_label_color=label_colors.get(link_label, "grey"),
                                    text=text,
                                    )
                        )
            j += 1

        start += length
    
    tree = create_tree(tree=TextNode(), nodes=nodes)
    tree.show()
                


#    def annotation2(self, line_pos, x, y):
#         X = [x] * len(line_pos)
#         Y = [y+pos+0.1 for _, pos in  line_pos]
#         text = [t for t,_ in line_pos]
#         return dict(
#                         x = X,
#                         y = Y,
#                         mode = "text",
#                         text = text,
#                         #texttemplate = "%{text}<br>(%{a:.2f}, %{b:.2f}, %{c:.2f})",
#                         #textposition = "center",
#                         textfont = {'family': "Times", 'size': 10, 'color': "DarkOrange"}
#                     )


    # def xgrid(self, width):
    #     num = 3 if width == 0 else (width*2) +1
    #     #grid = np.array([np.linspace(0, 100, num=num) for i in range(self.max_depth)])
    #     return np.linspace(0, 100, num=num)

    # def __max_width(self):
    #     width = len(self.children)
    #     childrens_width = 0
        
    #     level_width = len(self.children)
    #     for child in self.children:
    #         child_w = child.__max_width()
    #         child.level_width = level_width
    #         child.span = (childrens_width, child_w)
    #         childrens_width += child_w
        
    #     #self.level_width =  childrens_width
    #     if childrens_width > width:
    #         width = childrens_width
        
    #     self.__width_done = True
    #     self._max_width = width
    #     return width


# def get_color_hex(cmap_name:str, value=1.0):
#     norm = mpl.colors.Normalize(vmin=0.0,vmax=2)
#     cmap = cm.get_cmap(cmap_name)
#     hex_code = colors.to_hex(cmap(norm(value)))
#     return hex_code


# def normalize_row_coordinates(plot_data):
#     Y = np.array(plot_data["Y"])
#     X = np.array(plot_data["X"], dtype=float)

#     # where level is == 1, this is the level of all trees
#     # first part only alines the trees so they are equally distributed over the max width
#     tree_level_mask = Y==1
    
#     max_width = plot_data["max_width"]
#     max_depth = plot_data["max_depth"]
#     nr_trees = len(plot_data["tree_widths"])
#     if nr_trees == 1:
#         nr_trees += 1

#     new_xs = X[tree_level_mask] * ( max_width /   nr_trees)
#     X[tree_level_mask] = new_xs
#     plot_data["X"] = X.tolist()

#     for level in range(2, max_depth):
#         level_width = plot_data["level_widths"][level]
#         tree_level_mask = Y == level
#         new_xs = X[tree_level_mask] * ( max_width /   level_width)
#         X[tree_level_mask] = new_xs
#         plot_data["X"] = X.tolist()


# def set_link_coordinates(plot_data, labels:list):

#     for label in labels:
#         plot_data["link_xy"][label] = {
#                                         "X":[],
#                                         "Y": [],
#                                         }
        
#     for link_label, idx_of_parent, idx_of_child in plot_data["links"]:
#         plot_data["link_xy"][link_label]["X"].extend([plot_data["X"][idx_of_parent], plot_data["X"][idx_of_child], None])
#         plot_data["link_xy"][link_label]["Y"].extend([plot_data["Y"][idx_of_parent], plot_data["Y"][idx_of_child], None])


# def get_plot_data(data):

#     def place_node(tree_nodes, node, plot_data):
#         if node["link"] in tree_nodes:
#             parent_node = tree_nodes[node["link"]]
#             root = parent_node["root"]

#             row = parent_node["row"] + 1

#             if row not in plot_data["level_widths"]:
#                 plot_data["level_widths"][row] = 0
            
#             plot_data["level_widths"][row] += 1
#             column = plot_data["level_widths"][row]

#             idx = len(plot_data["X"])
#             plot_data["X"].append(column)
#             plot_data["Y"].append(row)
#             plot_data["root"].append(root)
#             plot_data["links"].append((node["link_label"], parent_node["idx"], idx))
#             plot_data["ids"].append(node["id"])
#             plot_data["labels"].append(node["label"])
#             plot_data["max_depth"] = max(plot_data["max_depth"], row)
#             plot_data["texts"].append(format_text(node["text"]))
#             parent_node["subnodes"][node["id"]] =  {
#                                                     "row":row,        
#                                                     "column":column,
#                                                     "root":root,
#                                                     "idx": idx,
#                                                     "subnodes":{}
#                                                     }
            
#             # we need to know the width of a tree so we can normalzie the coordinates later
#             plot_data["tree_widths"][root] = max(plot_data["tree_widths"][root], len(parent_node["subnodes"]))
#             return True
#         else:
#             for node_id, node_dict in tree_nodes.items():
#                 if place_node(node_dict["subnodes"], node, plot_data):
#                     return True
                    
#             return False

#     plot_data = {
#                 "X":[],
#                 "Y":[], 
#                 "links": [],
#                 "labels": [],
#                 "ids": [],
#                 "texts": [],
#                 "root":[], 
#                 "max_depth":0, 
#                 "tree_widths":{},
#                 "level_widths":{},
#                 "link_xy": {}
#                 }
#     trees = {}

#     # addin "seen" (and copy)
#     unplaced_nodes = [{**node,**{"seen":0}} for node in data.copy()]
#     while unplaced_nodes:

#         node  = unplaced_nodes.pop(0)

#         #if we have a node that points to itself its the root of a new tree
#         if node["id"] == node["link"]:
#             row = 1
#             column = len(trees)+1
#             idx = len(plot_data["X"])
#             trees[node["id"]] = {
#                             "row":row,        
#                             "column":column,
#                             "root":node["id"],
#                             "idx": idx,
#                             "subnodes":{}
#                             }
#             plot_data["X"].append(column)
#             plot_data["Y"].append(row)
#             plot_data["labels"].append(node["label"])
#             plot_data["root"].append(node["id"])
#             plot_data["ids"].append(node["id"])
#             plot_data["tree_widths"][node["id"]] = 1
#             plot_data["texts"].append(format_text(node["text"]))

#             if row not in plot_data["level_widths"]:
#                 plot_data["level_widths"][row] = 0
        
#             plot_data["level_widths"][row] += 1

#             found = True
#         else:
#             found = place_node(trees, node, plot_data)
    
#         # if node found a place we remove it else we just add it
#         # last to the unplaced_nodes list
#         if not found and node not in unplaced_nodes:

#             if node["seen"] >= 2:
#                 row = 1
#                 column = len(trees)+1
#                 idx = len(plot_data["X"])
#                 trees[node["id"]] = {
#                                 "row":row,        
#                                 "column":column,
#                                 "root":node["id"],
#                                 "idx": idx,
#                                 "subnodes":{}
#                                 }      
#                 plot_data["X"].append(column)
#                 plot_data["Y"].append(row)
#                 plot_data["root"].append(node["id"])
#                 plot_data["labels"].append(node["id"])
#                 plot_data["tree_widths"][node["id"]] = 1
#                 plot_data["texts"].append(format_text(node["text"]))

#                 if row not in plot_data["level_widths"]:
#                     plot_data["level_widths"][row] = 0
            
#                 plot_data["level_widths"][row] += 1

#             else:
#                 node["seen"] += 1
#                 unplaced_nodes.append(node)

#     plot_data["max_width"] =  max(plot_data["level_widths"].values())
#     return plot_data


# def add_node_text(fig, plot_data):
#     annotations = [dict(
#                                 text=l, 
#                                 x=plot_data["X"][i], 
#                                 y=plot_data["Y"][i],
#                                 xref='x1', yref='y1',
#                                 font=dict(color='black', size=10),
#                                 showarrow=False
#                                 ) 
#                     for i, l in enumerate(plot_data["ids"])]

#     fig.update_layout(
#                         annotations=annotations,
#                         )


# def replacement_legend(fig, legendgroup:str):
#     fig.add_trace(go.Scatter(
#                             x=[0],
#                             y=[0],
#                             visible=True,
#                             mode="markers",
#                             marker=dict(symbol='square',
#                                         size=50,
#                                         color="white"
#                                         ),
#                             name=legendgroup,
#                             showlegend=True,
#                             legendgroup=legendgroup,
#                             ))
#     fig.update_layout(
#                         legend= dict(
#                                     font=dict(
#                                                 family="Courier",
#                                                 size=15,
#                                                 color="black"
#                                              )
#                                     )
#                         )
    

# def add_lines(fig, plot_data, opacity:float, legendgroup:str, link_label2color:dict):
#     for label, link_data in plot_data["link_xy"].items():

#         legend = f"{legendgroup}-{label}"
#         style = dict(
#                     color=link_label2color[label], 
#                     width=2, 
#                     dash='solid'
#                     )

#         fig.add_trace(go.Scatter(
#                                 x=link_data["X"],
#                                 y=link_data["Y"],
#                                 mode='lines',
#                                 name=label,
#                                 line=style,
#                                 hoverinfo="none",
#                                 showlegend=True,
#                                 legendgroup=legendgroup,
#                                 opacity=opacity,
#                                 ))


# def add_nodes(fig, plot_data, opacity:float, legendgroup:str, node_size:int, label2color:dict):

#     groups = {l:{"X":[],"Y":[], "texts":[], "ids":[]} for l in label2color.keys()}
#     for x,y, text, ID, label in zip(plot_data["X"], plot_data["Y"], plot_data["texts"], plot_data["ids"], plot_data["labels"]):
#         groups[label]["X"].append(x)
#         groups[label]["Y"].append(y)
#         groups[label]["ids"].append(ID)
#         groups[label]["texts"].append(text)

#     for label, plot_data in groups.items():
#         legend = f"{legendgroup}-{label}"
#         fig.add_trace(go.Scatter(
#                                 x=plot_data["X"],
#                                 y=plot_data["Y"],
#                                 mode='markers+text',
#                                 name=label,
#                                 legendgroup=legendgroup,
#                                 showlegend=True,
#                                 marker=dict(    
#                                                 #symbol='diamond-wide',
#                                                 symbol='circle',
#                                                 size=node_size,
#                                                 color=label2color[label],
#                                                 #opacity=colors,
#                                                 ),
#                                 text=plot_data["ids"],
#                                 hovertext=plot_data["texts"],
#                                 hoverinfo='text',
#                                 opacity=opacity,
#                                 textposition='middle center'
                                
#                                 ))


# def format_text(text:str):
#     tokens = text.split(" ")
#     return "<br>".join([" ".join(tokens[i:i+10]) for i in range(0,len(tokens),10)])


# def convert_links_ids(data):
#     """
#     changes link from int to string id
#     """
#     id2idx = {d["id"]:i for i,d in enumerate(data)}

#     for node in data:
#         link  = node["link"]

#         if isinstance(link, int):
#             node["link_int"] = link
#             node["link"] = data[link]["id"]
#         else:
#             node["link_int"] = id2idx[link]


# def create_ids(data):
#     node_stack = {}
#     for node in data:
#         label = node["label"]

#         if label not in node_stack:
#             node_stack[label] = []
        
#         node_id = f"{label}_{len(node_stack[label])}"
        
#         node_stack[label].append(label)
#         node["id"] = node_id


# def create_tree_plot(fig, 
#                     data:dict,
#                     label2color:dict,
#                     link_label2color:dict,                      
#                     reverse:bool, 
#                     opacity:float=1.0, 
#                     legendgroup:str=None, 
#                     node_size:int=90
#                     ): # nodes:list, links:list, color:str):

#     data = deepcopy(data)

#     #we turn labels into ids
#     if "id" not in data[0]:
#         create_ids(data)

#     # change links
#     convert_links_ids(data)

#     # sort data to untangle trees
#     data = sorted(data, key=lambda x:x["link_int"])

#     # parse the data to get plot data ( e.g. X Y coordinates, max depth, max width etc)
#     plot_data = get_plot_data(data)

#     #normalize coordinates to make tree abit nicer now as we know the depth and widths
#     normalize_row_coordinates(plot_data)

#     # when we have normalized X and Y corridnates we can set up all the links
#     link_labels = list(link_label2color.keys())
#     set_link_coordinates(plot_data, labels=link_labels)
    
#     #create the plot
#     add_lines(fig, plot_data, opacity=opacity, legendgroup=legendgroup, link_label2color=link_label2color)
#     add_nodes(fig, plot_data, opacity=opacity, legendgroup=legendgroup, node_size=node_size, label2color=label2color)
#     #add_node_text(fig, plot_data)
    
#     return plot_data["max_depth"], plot_data["max_width"]


# def get_colors(data, gold_data, labels, link_labels, label2color, link_label2color):


#     if not labels:
#         labels = set([d["label"] for d in data])

#         if gold_data:
#             labels.update([d["label"] for d in gold_data])
        
#         labels = sorted(labels)

#     if not link_labels:
        
#         link_labels = set([d["link_label"] for d in data])

#         if gold_data:
#             link_labels.update([d["link_label"] for d in gold_data])

#         link_labels = sorted(link_labels)

#     if labels and not label2color:
#         lc = ["#5ac18e", "#7fe5f0", "#ffa500", "#008080", "#ff4040", "#8a2be2", "#b6fcd5", "#b4eeb4", "#ff7f50"]
#         label2color = {l:lc[i] for i,l in enumerate(labels)}

#     if link_labels and not link_label2color:
#         rc = ["#e43518", "#25c130", "#7c1dbb", "#998002", "#0f34d4", "#f58a00"]

#         link_label2color = {r:rc[i] for i,r in enumerate(link_labels)}

#     return label2color, link_label2color


# def hot_tree(
    #         data, 
    #         gold_data=None, 
    #         labels:list=None, 
    #         link_labels:list=None, 
    #         reverse:bool=True, 
    #         title:str="", 
    #         save_to:str=None, 
    #         node_size=90,
    #         label2color:dict=None,
    #         link_label2color:dict=None,
    #         ):

    # label2color, link_label2color = get_colors(data, gold_data, labels, link_labels, label2color, link_label2color)

    # fig = go.Figure()   

    # # gold data is added , we can create a tree with high opacity that just sits static in the background
    # if gold_data:
    #     replacement_legend(fig, "gold")
    #     gold_max_depth, gold_max_width = create_tree_plot(  
    #                                                         fig, 
    #                                                         data=gold_data,
    #                                                         label2color=label2color,
    #                                                         link_label2color=link_label2color,
    #                                                         reverse=reverse,
    #                                                         opacity=0.3,
    #                                                         legendgroup="gold",
    #                                                         node_size=node_size
    #                                                         )

    # legendgroup = None
    # if gold_data:
    #     replacement_legend(fig, "pred")
    #     legendgroup = "pred"
    
    # max_depth, max_width = create_tree_plot(  
    #                                         fig, 
    #                                         data=data,
    #                                         label2color=label2color,
    #                                         link_label2color=link_label2color,                                            
    #                                         reverse=reverse,
    #                                         legendgroup=legendgroup,
    #                                         node_size=node_size
    #                                         )  

    # if gold_data:
    #     max_depth = max(gold_max_depth, max_depth) 
    #     max_width = max(gold_max_width, max_width)

    # fig.update_layout(
    #                 yaxis=dict(range=[0,max_depth+1], autorange="reversed" if reverse else None),
    #                 xaxis=dict(range=[0,max_width+1])
    #                 )
    

    # axis = dict(
    #             showline=False, # hide axis line, grid, ticklabels and  title
    #             zeroline=False,
    #             showgrid=False,
    #             showticklabels=False,
    #         )

    # fig.update_layout(
    #                     title=title,
    #                     #annotations=make_annotations(name2pos),
    #                     font_size=12,
    #                     showlegend=True,
    #                     margin=dict(l=40, r=40, b=85, t=100),
    #                     hovermode='closest',
    #                     plot_bgcolor="white", #'rgb(248,248,248)',
    #                     xaxis=axis,
    #                     yaxis=axis
    #                     )
    
    # if save_to:
    #     fig.write_image(save_to)

    # return fig