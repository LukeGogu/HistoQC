import logging
import os

import math
from distutils.util import strtobool
import skimage
from histoqc.BaseImage import printMaskHelper
from skimage import io, img_as_ubyte, morphology, measure
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import rank
import numpy as np

from enum import Enum
from PIL import Image, ImageDraw, ImageFont

def generateScaledMaskedImage(s, params):
    
    img = s.getImgThumb(params.get("image_work_size", "1.25x"))
    mask = skimage.transform.resize(s["img_mask_use"], img[:, :, 1].shape, order=0) > 0
    masked_img = img*np.dstack([mask, mask, mask])
    io.imsave(s["outdir"] + os.sep + s["filename"] + "_scaled_mask_applied.png", img_as_ubyte(masked_img))
    
    return

def maskedImage(img, mask):
    masked_img = img*np.dstack([mask, mask, mask])
    return masked_img




def extractTiles(slide, params):

    logging.info(f"{slide['filename']} - \textractTissueTiles")
    
    global ROW_TILE_SIZE
    global COL_TILE_SIZE
    global ROW_OVERLAP
    global COL_OVERLAP
    global TISSUE_HIGH_THRESH
    global TISSUE_LOW_THRESH
    global HSV_PURPLE
    global HSV_PINK
    global NUM_TOP_TILES
    global MAX_TILE_NUM
    global TILE_SAVE_LEVEL
    global s

    ROW_TILE_SIZE = int(params.get("row_size", 128))
    COL_TILE_SIZE = int(params.get("col_size", ROW_TILE_SIZE))
    ROW_TILE_SIZE= int(params.get("patchsize", ROW_TILE_SIZE))
    COL_TILE_SIZE = int(params.get("patchsize", COL_TILE_SIZE))
    
    ROW_OVERLAP = int(params.get("row_overlap", 0))
    COL_OVERLAP = int(params.get("col_overlap", ROW_OVERLAP))
    ROW_OVERLAP = int(params.get("overlap", ROW_OVERLAP))
    COL_OVERLAP = int(params.get("overlap", COL_OVERLAP))

    TISSUE_HIGH_THRESH = int(params.get("tissue_high_thresh", 80))
    TISSUE_LOW_THRESH = int(params.get("tissue_low_thresh", 10))

    HSV_PURPLE = int(params.get("hsv_purple", 270))
    HSV_PINK = int(params.get("hsv_pink", 330))

    MAX_TILE_NUM = int(params.get("max_tile_num", 1000))            #-1 means all
    NUM_TOP_TILES = int(params.get("num_top_tiles", min(50, MAX_TILE_NUM)))

    TILE_SAVE_LEVEL = params.get("image_work_size", "10.0x")

    save_tiles_with_tissue = strtobool(params.get("tissue_tiles", "False"))
    save_top_tiles = strtobool(params.get("top_tiles", str(not save_tiles_with_tissue)))
    save_csv = strtobool(params.get("save_csv", "False"))
    
    s = slide

    summary_and_tiles(display=False, save_summary=True, save_data=save_csv, save_top_tiles=save_top_tiles, save_tiles_with_tissue=save_tiles_with_tissue)



DISPLAY_TILE_SUMMARY_LABELS = False
TILE_LABEL_TEXT_SIZE = 10
LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY = True
BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY = True

TILE_BORDER_SIZE = 2  # The size of the colored rectangular border around summary tiles.

HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

FADED_THRESH_COLOR = (128, 255, 128)
FADED_MEDIUM_COLOR = (255, 255, 128)
FADED_LOW_COLOR = (255, 210, 128)
FADED_NONE_COLOR = (255, 128, 128)

FONT_PATH = "C:/Windows/Fonts/arialbd.ttf"
SUMMARY_TITLE_FONT_PATH = "C:/Windows/Fonts/courbd.ttf"
SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 24
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4


def get_num_tiles(h, w):
  """
  Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
  a column tile size.

  Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
    into given the row tile size and the column tile size.
  """
  num_row_tiles = math.floor((h - ROW_OVERLAP)/ (ROW_TILE_SIZE - ROW_OVERLAP))
  num_col_tiles = math.floor((w - COL_OVERLAP)/ (COL_TILE_SIZE - ROW_OVERLAP))
  return num_row_tiles, num_col_tiles


def get_tile_indices(h, w):
  """
  Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).

  Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    List of tuples representing tile coordinates consisting of starting row, ending row,
    starting column, ending column, row number, column number.
  """
  indices = list()
  num_row_tiles, num_col_tiles = get_num_tiles(h, w)
  for r in range(0, num_row_tiles):
    start_r = r * (ROW_TILE_SIZE - ROW_OVERLAP)
    end_r = start_r + ROW_TILE_SIZE #if (r < num_row_tiles - 1) else h
    for c in range(0, num_col_tiles):
      start_c = c * (COL_TILE_SIZE - COL_OVERLAP)
      end_c = start_c + COL_TILE_SIZE #if (c < num_col_tiles - 1) else w
      indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
  return indices


def summary_and_tiles(display=True, save_summary=False, save_data=True, save_top_tiles=True, save_tiles_with_tissue=False):
  """
  Generate tile summary and top tiles for slide.

  Args:
    slide_num: The slide number.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.

  """
  slide_name = s["filename"]
  img = s.getImgThumb(s["image_work_size"])
  mask = s["img_mask_use"]
  np_img = maskedImage(img, mask)

  tile_sum = score_tiles(np_img)
  dir=os.path.join(s["outdir"], "tiles")
  if not os.path.exists(dir):
      os.makedirs(dir)
  if save_data:
    save_tile_data(tile_sum)
  generate_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  generate_top_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  generate_tissue_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  if save_top_tiles:
    for tile in tile_sum.top_tiles():
      tile.save_tile("top_tiles")
  if save_tiles_with_tissue:
    for tile in tile_sum.top_tiles_tissue_thresh():
        tile.save_tile("tissue_tiles")
  return tile_sum


def save_tile_as_png(tile, save=True, folder=""):
  """
  Save and/or display a tile image.

  Args:
    tile: Tile object.
    save: If True, save tile image.
    display: If True, dispaly tile image.
  """
  t = tile
  np_tile = s.getImgThumb(TILE_SAVE_LEVEL)[t.o_r_s:t.o_r_e, t.o_c_s:t.o_c_e]

  if save:
    img_path = os.path.join(s["outdir"], "tiles", folder, "tile-r%d-c%d-x%d-y%d-w%d-h%d" % (
                             t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s) + ".png")
    dir = os.path.dirname(img_path)
    if not os.path.exists(dir):
      os.makedirs(dir)
    io.imsave(img_path, img_as_ubyte(np_tile))


def score_tiles(masked_img, small_tile_in_tile=False):
  """
  Score all tiles for a slide and return the results in a TileSummary object.

  Args:
    slide_num: The slide number.
    np_img: Optional image as a NumPy array.
    dimensions: Optional tuple consisting of (original width, original height, new width, new height). Used for dynamic
      tile retrieval.
    small_tile_in_tile: If True, include the small NumPy image in the Tile objects.

  Returns:
    TileSummary object which includes a list of Tile objects containing information about each tile.
  """
  tile_extr_img = s.getImgThumb(TILE_SAVE_LEVEL)

  (h, w, _) = masked_img.shape
  (o_h, o_w, _) = tile_extr_img.shape

  scale_factor = max(masked_img.shape)/max(tile_extr_img.shape)

  row_tile_size = round(ROW_TILE_SIZE * scale_factor)  # use round?
  col_tile_size = round(COL_TILE_SIZE * scale_factor)  # use round?

  num_row_tiles, num_col_tiles = get_num_tiles(o_h, o_w)

  tile_sum = TileSummary(orig_w=o_w,
                         orig_h=o_h,
                         orig_tile_w=COL_TILE_SIZE,
                         orig_tile_h=ROW_TILE_SIZE,
                         scale_factor=scale_factor,
                         scaled_w=w,
                         scaled_h=h,
                         scaled_tile_w=col_tile_size,
                         scaled_tile_h=row_tile_size,
                         tissue_percentage=100-mask_percent(masked_img),
                         num_col_tiles=num_col_tiles,
                         num_row_tiles=num_row_tiles)

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0
  sf = scale_factor
  tile_indices = get_tile_indices(o_h, o_w)
  scaled_tile_indices = tuple(np.rint(np.array(tile_indices) * np.array((sf,sf,sf,sf,1,1))).astype(int))
  for st, t in zip(scaled_tile_indices, tile_indices):
    count += 1  # tile_num
    r_s, r_e, c_s, c_e, r, c = st
    np_tile = masked_img[r_s:r_e, c_s:c_e]
    t_p = tissue_percent(np_tile)
    amount = tissue_quantity(t_p)
    if amount == TissueQuantity.HIGH:
      high += 1
    elif amount == TissueQuantity.MEDIUM:
      medium += 1
    elif amount == TissueQuantity.LOW:
      low += 1
    elif amount == TissueQuantity.NONE:
      none += 1
    o_r_s, o_r_e, o_c_s, o_c_e, _, _ = t

    score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_tile, t_p, r, c)

    np_scaled_tile = np_tile if small_tile_in_tile else None
    tile = Tile(tile_sum, np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score)
    tile_sum.tiles.append(tile)

  tile_sum.count = count
  tile_sum.high = high
  tile_sum.medium = medium
  tile_sum.low = low
  tile_sum.none = none

  tiles_by_score = tile_sum.tiles_by_score()
  rank = 0
  for t in tiles_by_score:
    rank += 1
    t.rank = rank

  return tile_sum


class TileSummary:
  """
  Class for tile summary information.
  """

  orig_w = None
  orig_h = None
  orig_tile_w = None
  orig_tile_h = None
  scaled_w = None
  scaled_h = None
  scaled_tile_w = None
  scaled_tile_h = None
  mask_percentage = None
  num_row_tiles = None
  num_col_tiles = None

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0

  def __init__(self, orig_w, orig_h, orig_tile_w, orig_tile_h, scale_factor, scaled_w, scaled_h, scaled_tile_w,
               scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles):
    self.slide_name = s["filename"]
    self.orig_w = orig_w
    self.orig_h = orig_h
    self.orig_tile_w = orig_tile_w
    self.orig_tile_h = orig_tile_h
    self.scale_factor = scale_factor
    self.scaled_w = scaled_w
    self.scaled_h = scaled_h
    self.scaled_tile_w = scaled_tile_w
    self.scaled_tile_h = scaled_tile_h
    self.tissue_percentage = tissue_percentage
    self.num_col_tiles = num_col_tiles
    self.num_row_tiles = num_row_tiles
    self.tiles = []

  def __str__(self):
    return summary_title(self) + "\n" + summary_stats(self)

  def mask_percentage(self):
    """
    Obtain the percentage of the slide that is masked.

    Returns:
       The amount of the slide that is masked as a percentage.
    """
    return 100 - self.tissue_percentage

  def num_tiles(self):
    """
    Retrieve the total number of tiles.

    Returns:
      The total number of tiles (number of rows * number of columns).
    """
    return self.num_row_tiles * self.num_col_tiles

  def tiles_by_tissue_percentage(self):
    """
    Retrieve the tiles ranked by tissue percentage.

    Returns:
       List of the tiles ranked by tissue percentage.
    """
    sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
    return sorted_list

  def tiles_by_score(self):
    """
    Retrieve the tiles ranked by score.

    Returns:
       List of the tiles ranked by score.
    """
    sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
    return sorted_list

  def top_tiles(self):
    """
    Retrieve the top-scoring tiles.

    Returns:
       List of the top-scoring tiles.
    """
    sorted_tiles = self.tiles_by_score()
    top_tiles = sorted_tiles[:NUM_TOP_TILES]
    return top_tiles

  def top_tiles_tissue_thresh(self):

    sorted_list = self.tiles_by_score()
    sorted_list_thres = list(filter(lambda t: t.tissue_percentage >= TISSUE_HIGH_THRESH, sorted_list))
    if MAX_TILE_NUM == -1 or MAX_TILE_NUM>len(sorted_list_thres):
        i = len(sorted_list_thres)
    else: i = MAX_TILE_NUM
    top_tiles_tissue_thresh = sorted_list_thres[:i]
    return top_tiles_tissue_thresh

  def tiles_tissue_thresh(self):

    list_thresh = list(filter(lambda t: t.tissue_percentage >= TISSUE_HIGH_THRESH, self.tiles()))
    return list_thresh

  def get_tile(self, row, col):
    """
    Retrieve tile by row and column.

    Args:
      row: The row
      col: The column

    Returns:
      Corresponding Tile object.
    """
    tile_index = (row - 1) * self.num_col_tiles + (col - 1)
    tile = self.tiles[tile_index]
    return tile

  def display_summaries(self):
    """
    Display summary images.
    """
    summary_and_tiles(display=True, save_summary=False, save_data=False, save_top_tiles=False)


class Tile:
  """
  Class for information about a tile.
  """

  def __init__(self, tile_summary, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
               o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score):
    self.tile_summary = tile_summary
    self.slide_name = s["filename"]
    self.np_scaled_tile = np_scaled_tile
    self.tile_num = tile_num
    self.r = r
    self.c = c
    self.r_s = r_s
    self.r_e = r_e
    self.c_s = c_s
    self.c_e = c_e
    self.o_r_s = o_r_s
    self.o_r_e = o_r_e
    self.o_c_s = o_c_s
    self.o_c_e = o_c_e
    self.tissue_percentage = t_p
    self.color_factor = color_factor
    self.s_and_v_factor = s_and_v_factor
    self.quantity_factor = quantity_factor
    self.score = score

  def __str__(self):
    return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
      self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

  def __repr__(self):
    return "\n" + self.__str__()

  def mask_percentage(self):
    return 100 - self.tissue_percentage

  def tissue_quantity(self):
    return tissue_quantity(self.tissue_percentage)

  def get_pil_tile(self):
    return tile_to_pil_tile(self)

  def get_np_tile(self):
    return tile_to_np_tile(self)

  def save_tile(self, folder):
    save_tile_as_png(self, save=True, folder=folder)

  #def display_tile(self):
  #  save_display_tile(self, save=False, display=True)

  def display_with_histograms(self):
    display_tile(self, rgb_histograms=True, hsv_histograms=True)

  def get_np_scaled_tile(self):
    return self.np_scaled_tile

  def get_pil_scaled_tile(self):
    return np_to_pil(self.np_scaled_tile)


def score_tile(np_tile, tissue_percent, row, col):
  """
  Score tile based on tissue percentage, color factor, saturation/value factor, and tissue quantity factor.

  Args:
    np_tile: Tile as NumPy array.
    tissue_percent: The percentage of the tile judged to be tissue.
    slide_num: Slide number.
    row: Tile row.
    col: Tile column.

  Returns tuple consisting of score, color factor, saturation/value factor, and tissue quantity factor.
  """
  color_factor = hsv_purple_pink_factor(np_tile)
  s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
  amount = tissue_quantity(tissue_percent)
  quantity_factor = tissue_quantity_factor(amount)
  combined_factor = color_factor * s_and_v_factor * quantity_factor
  score = (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0
  # scale score to between 0 and 1
  score = 1.0 - (10.0 / (10.0 + score))
  return score, color_factor, s_and_v_factor, quantity_factor

def hsv_purple_pink_factor(rgb):
  """
  Compute scoring factor based on purple and pink HSV hue deviations and degree to which a narrowed hue color range
  average is purple versus pink.

  Args:
    rgb: Image an NumPy array.

  Returns:
    Factor that favors purple (hematoxylin stained) tissue over pink (eosin stained) tissue.
  """
  hues = rgb_to_hues(rgb)
  hues = hues[hues >= 260]  # exclude hues under 260
  hues = hues[hues <= 340]  # exclude hues over 340
  if len(hues) == 0:
    return 0  # if no hues between 260 and 340, then not purple or pink
  pu_dev = hsv_purple_deviation(hues)
  pi_dev = hsv_pink_deviation(hues)
  avg_factor = (340 - np.average(hues)) ** 2

  if pu_dev == 0:  # avoid divide by zero if tile has no tissue
    return 0

  factor = pi_dev / pu_dev * avg_factor
  return factor

def hsv_purple_deviation(hsv_hues):
  """
  Obtain the deviation from the HSV hue for purple.

  Args:
    hsv_hues: NumPy array of HSV hue values.

  Returns:
    The HSV purple deviation.
  """
  purple_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PURPLE) ** 2))
  return purple_deviation


def hsv_pink_deviation(hsv_hues):
  """
  Obtain the deviation from the HSV hue for pink.

  Args:
    hsv_hues: NumPy array of HSV hue values.

  Returns:
    The HSV pink deviation.
  """
  pink_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PINK) ** 2))
  return pink_deviation

def hsv_saturation_and_value_factor(rgb):
  """
  Function to reduce scores of tiles with narrow HSV saturations and values since saturation and value standard
  deviations should be relatively broad if the tile contains significant tissue.

  Example of a blurred tile that should not be ranked as a top tile:
    ../data/tiles_png/006/TUPAC-TR-006-tile-r58-c3-x2048-y58369-w1024-h1024.png

  Args:
    rgb: RGB image as a NumPy array

  Returns:
    Saturation and value factor, where 1 is no effect and less than 1 means the standard deviations of saturation and
    value are relatively small.
  """
  hsv = rgb2hsv(rgb)
  s = filter_hsv_to_s(hsv)
  v = filter_hsv_to_v(hsv)
  s_std = np.std(s)
  v_std = np.std(v)
  if s_std < 0.05 and v_std < 0.05:
    factor = 0.4
  elif s_std < 0.05:
    factor = 0.7
  elif v_std < 0.05:
    factor = 0.7
  else:
    factor = 1

  factor = factor ** 2
  return factor

def rgb_to_hues(rgb):
  """
  Convert RGB NumPy array to 1-dimensional array of hue values (HSV H values in degrees).

  Args:
    rgb: RGB image as a NumPy array

  Returns:
    1-dimensional array of hue values in degrees
  """
  hsv = rgb2hsv(rgb)
  h = filter_hsv_to_h(hsv)
  return h

def filter_hsv_to_h(hsv, output_type="int"):
  """
  Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
  values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
  https://en.wikipedia.org/wiki/HSL_and_HSV

  Args:
    hsv: HSV image as a NumPy array.
    output_type: Type of array to return (float or int).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    Hue values (float or int) as a 1-dimensional NumPy array.
  """
  h = hsv[:, :, 0]
  h = h.flatten()
  if output_type == "int":
    h *= 360
    h = h.astype("int")
  return h

def filter_hsv_to_s(hsv):
  """
  Experimental HSV to S (saturation).

  Args:
    hsv:  HSV image as a NumPy array.

  Returns:
    Saturation values as a 1-dimensional NumPy array.
  """
  s = hsv[:, :, 1]
  s = s.flatten()
  return s

def filter_hsv_to_v(hsv):
  """
  Experimental HSV to V (value).

  Args:
    hsv:  HSV image as a NumPy array.

  Returns:
    Value values as a 1-dimensional NumPy array.
  """
  v = hsv[:, :, 2]
  v = v.flatten()
  return v

def mask_percent(np_img):
  """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is masked.
  """
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage

def tissue_percent(np_img):
  """
  Determine the percentage of a NumPy array that is tissue (not masked).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is tissue.
  """
  return 100 - mask_percent(np_img)

def tissue_quantity_factor(amount):
  """
  Obtain a scoring factor based on the quantity of tissue in a tile.

  Args:
    amount: Tissue amount as a TissueQuantity enum value.

  Returns:
    Scoring factor based on the tile tissue quantity.
  """
  if amount == TissueQuantity.HIGH:
    quantity_factor = 1.0
  elif amount == TissueQuantity.MEDIUM:
    quantity_factor = 0.2
  elif amount == TissueQuantity.LOW:
    quantity_factor = 0.1
  else:
    quantity_factor = 0.0
  return quantity_factor

def tissue_quantity(tissue_percentage):
  """
  Obtain TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE) for corresponding tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage.

  Returns:
    TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE).
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    return TissueQuantity.HIGH
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    return TissueQuantity.MEDIUM
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    return TissueQuantity.LOW
  else:
    return TissueQuantity.NONE

class TissueQuantity(Enum):
  NONE = 0
  LOW = 1
  MEDIUM = 2
  HIGH = 3

def generate_tile_summaries(tile_sum, np_img, display=True, save_summary=False):
  """
  Generate summary images/thumbnails showing a 'heatmap' representation of the tissue segmentation of all tiles.

  Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
  """
  z = 300  # height of area at top of summary slide
  slide_name = tile_sum.slide_name
  rows = tile_sum.orig_h
  cols = tile_sum.orig_w
  row_tile_size = tile_sum.scaled_tile_h
  col_tile_size = tile_sum.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  np_orig = s.getImgThumb(s["image_work_size"])
  summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw_orig = ImageDraw.Draw(summary_orig)

  for t in tile_sum.tiles:
    border_color = tile_border_color(t.tissue_percentage)
    tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)

  summary_txt = "%s Tile Summary:" % tile_sum.slide_name + "\n" + summary_stats(tile_sum)

  summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
  draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
  draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

  if DISPLAY_TILE_SUMMARY_LABELS:
    count = 0
    for t in tile_sum.tiles:
      count += 1
      label = "R%d\nC%d" % (t.r, t.c)
      font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
      # drop shadow behind text
      draw.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)
      draw_orig.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)

      draw.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
      draw_orig.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

  if display:
    summary.show()
    summary_orig.show()
  if save_summary:
    summary.save(s["outdir"] + os.sep + "tiles" + os.sep + s["filename"] + "_tile_sum_masked.png")#save_tile_summary_image(summary, slide_num)
    summary_orig.save(s["outdir"] + os.sep + "tiles" + os.sep + s["filename"] + "_tile_sum_orig.png")#save_tile_summary_on_original_image(summary_orig, slide_num)

def generate_top_tile_summaries(tile_sum, np_img, display=True, save_summary=False, show_top_stats=True,
                                label_all_tiles=LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY,
                                border_all_tiles=BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY):
  """
  Generate summary images/thumbnails showing the top tiles ranked by score.

  Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display top tiles to screen.
    save_summary: If True, save top tiles images.
    show_top_stats: If True, append top tile score stats to image.
    label_all_tiles: If True, label all tiles. If False, label only top tiles.
  """
  z = 300  # height of area at top of summary slide
  slide_name = tile_sum.slide_name
  rows = tile_sum.orig_h
  cols = tile_sum.orig_w
  row_tile_size = tile_sum.scaled_tile_h
  col_tile_size = tile_sum.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  np_orig = s.getImgThumb(s["image_work_size"])
  summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw_orig = ImageDraw.Draw(summary_orig)

  if border_all_tiles:
    for t in tile_sum.tiles:
      border_color = faded_tile_border_color(t.tissue_percentage)
      tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)
      tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)

  tbs = TILE_BORDER_SIZE
  top_tiles = tile_sum.top_tiles()
  for t in top_tiles:
    border_color = tile_border_color(t.tissue_percentage)
    tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    if border_all_tiles:
      tile_border(draw, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))
      tile_border(draw_orig, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))

  summary_txt = "%s Top Tile Summary:" % tile_sum.slide_name + "\n" + summary_stats(tile_sum)

  summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
  draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
  draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

  tiles_to_label = tile_sum.tiles if label_all_tiles else top_tiles
  h_offset = TILE_BORDER_SIZE + 2
  v_offset = TILE_BORDER_SIZE
  h_ds_offset = TILE_BORDER_SIZE + 3
  v_ds_offset = TILE_BORDER_SIZE + 1
  for t in tiles_to_label:
    label = "R%d\nC%d" % (t.r, t.c)
    font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
    # drop shadow behind text
    draw.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)
    draw_orig.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)

    draw.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
    draw_orig.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

  if show_top_stats:
    summary = add_tile_stats_to_top_tile_summary(summary, top_tiles, z)
    summary_orig = add_tile_stats_to_top_tile_summary(summary_orig, top_tiles, z)

  if display:
    summary.show()
    summary_orig.show()
  if save_summary:
    summary.save(s["outdir"] + os.sep + "tiles" + os.sep + s["filename"] + "_top_tiles_masked.png")#save_top_tiles_image(summary, slide_num)
    summary_orig.save(s["outdir"] + os.sep + "tiles" + os.sep + s["filename"] + "_top_tiles_orig.png")#save_top_tiles_on_original_image(summary_orig, slide_num)
    
def generate_tissue_tile_summaries(tile_sum, np_img, display=True, save_summary=False, show_top_stats=False,
                                label_all_tiles=LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY,
                                border_all_tiles=BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY):
  """
  Generate summary images/thumbnails showing the top tiles ranked by score.

  Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display top tiles to screen.
    save_summary: If True, save top tiles images.
    show_top_stats: If True, append top tile score stats to image.
    label_all_tiles: If True, label all tiles. If False, label only top tiles.
  """
  z = 300  # height of area at top of summary slide
  slide_name = tile_sum.slide_name
  rows = tile_sum.orig_h
  cols = tile_sum.orig_w
  row_tile_size = tile_sum.scaled_tile_h
  col_tile_size = tile_sum.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  np_orig = s.getImgThumb(s["image_work_size"])
  summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw_orig = ImageDraw.Draw(summary_orig)

  if border_all_tiles:
    for t in tile_sum.tiles:
      border_color = faded_tile_border_color(t.tissue_percentage)
      tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)
      tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)

  tbs = TILE_BORDER_SIZE
  top_tiles = tile_sum.top_tiles_tissue_thresh()
  for t in top_tiles:
    border_color = tile_border_color(t.tissue_percentage)
    tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    if border_all_tiles:
      tile_border(draw, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))
      tile_border(draw_orig, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))

  summary_txt = "%s Tissue Tile Summary:" % tile_sum.slide_name + "\n" + summary_stats(tile_sum)

  summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
  draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
  draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

  tiles_to_label = tile_sum.tiles if label_all_tiles else top_tiles
  h_offset = TILE_BORDER_SIZE + 2
  v_offset = TILE_BORDER_SIZE
  h_ds_offset = TILE_BORDER_SIZE + 3
  v_ds_offset = TILE_BORDER_SIZE + 1
  for t in tiles_to_label:
    label = "R%d\nC%d" % (t.r, t.c)
    font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
    # drop shadow behind text
    draw.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)
    draw_orig.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)

    draw.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
    draw_orig.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

  if show_top_stats:
    summary = add_tile_stats_to_top_tile_summary(summary, top_tiles, z)
    summary_orig = add_tile_stats_to_top_tile_summary(summary_orig, top_tiles, z)

  if display:
    summary.show()
    summary_orig.show()
  if save_summary:
    summary.save(s["outdir"] + os.sep + "tiles" + os.sep + s["filename"] + "_tissue_tiles_masked.png")#save_top_tiles_image(summary, slide_num)
    summary_orig.save(s["outdir"] + os.sep + "tiles" + os.sep + s["filename"] + "_tissue_tiles_orig.png")#save_top_tiles_on_original_image(summary_orig, slide_num)

def create_summary_pil_img(np_img, title_area_height, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles):
  """
  Create a PIL summary image including top title area and right side and bottom padding.

  Args:
    np_img: Image as a NumPy array.
    title_area_height: Height of the title area at the top of the summary image.
    row_tile_size: The tile size in rows.
    col_tile_size: The tile size in columns.
    num_row_tiles: The number of row tiles.
    num_col_tiles: The number of column tiles.

  Returns:
    Summary image as a PIL image. This image contains the image data specified by the np_img input and also has
    potentially a top title area and right side and bottom padding.
  """
  r = np_img.shape[0] + title_area_height
  c = np_img.shape[1]
  summary_img = np.zeros([r, c, np_img.shape[2]], dtype=np.uint8)
  # add gray edges so that tile text does not get cut off
  summary_img.fill(120)
  # color title area white
  summary_img[0:title_area_height, 0:summary_img.shape[1]].fill(255)
  summary_img[title_area_height:np_img.shape[0] + title_area_height, 0:np_img.shape[1]] = np_img
  summary = np_to_pil(summary_img)
  return summary

def summary_stats(tile_summary):
  """
  Obtain various stats about the slide tiles.

  Args:
    tile_summary: TileSummary object.

  Returns:
     Various stats about the slide tiles as a string.
  """
  return "Tile Extraction Dimensions: %dx%d\n" % (tile_summary.orig_w, tile_summary.orig_h) + \
         "Original Tile Size: %dx%d\n" % (tile_summary.orig_tile_w, tile_summary.orig_tile_h) + \
         "Scale Factor: %3.2fx\n" % tile_summary.scale_factor + \
         "Scaled Dimensions: %dx%d\n" % (tile_summary.scaled_w, tile_summary.scaled_h) + \
         "Scaled Tile Size: %dx%d\n" % (tile_summary.scaled_tile_w, tile_summary.scaled_tile_w) + \
         "Total Mask: %3.2f%%, Total Tissue: %3.2f%%\n" % (
           tile_summary.mask_percentage(), tile_summary.tissue_percentage) + \
         "Tiles: %dx%d = %d\n" % (tile_summary.num_col_tiles, tile_summary.num_row_tiles, tile_summary.count) + \
         " %5d (%5.2f%%) tiles >=%d%% tissue\n" % (
           tile_summary.high, tile_summary.high / tile_summary.count * 100, TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >=%d%% and <%d%% tissue\n" % (
           tile_summary.medium, tile_summary.medium / tile_summary.count * 100, TISSUE_LOW_THRESH,
           TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >0%% and <%d%% tissue\n" % (
           tile_summary.low, tile_summary.low / tile_summary.count * 100, TISSUE_LOW_THRESH) + \
         " %5d (%5.2f%%) tiles =0%% tissue" % (tile_summary.none, tile_summary.none / tile_summary.count * 100)

def np_to_pil(np_img):
  """
  Convert a NumPy array to a PIL Image.

  Args:
    np_img: The image represented as a NumPy array.

  Returns:
     The NumPy array converted to a PIL Image.
  """
  if np_img.dtype == "bool":
    np_img = np_img.astype("uint8") * 255
  elif np_img.dtype == "float64":
    np_img = (np_img * 255).astype("uint8")
  return Image.fromarray(np_img)

def tile_border_color(tissue_percentage):
  """
  Obtain the corresponding tile border color for a particular tile tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage

  Returns:
    The tile border color corresponding to the tile tissue percentage.
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    border_color = HIGH_COLOR
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    border_color = MEDIUM_COLOR
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    border_color = LOW_COLOR
  else:
    border_color = NONE_COLOR
  return border_color

def tile_border(draw, r_s, r_e, c_s, c_e, color, border_size=TILE_BORDER_SIZE):
  """
  Draw a border around a tile with width TILE_BORDER_SIZE.

  Args:
    draw: Draw object for drawing on PIL image.
    r_s: Row starting pixel.
    r_e: Row ending pixel.
    c_s: Column starting pixel.
    c_e: Column ending pixel.
    color: Color of the border.
    border_size: Width of tile border in pixels.
  """
  for x in range(0, border_size):
    draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)

def add_tile_stats_to_top_tile_summary(pil_img, tiles, z):
  np_sum = np.asarray(pil_img)
  sum_r, sum_c, sum_ch = np_sum.shape
  np_stats = np_tile_stat_img(tiles)
  st_r, st_c, _ = np_stats.shape
  combo_c = sum_c + st_c
  combo_r = max(sum_r, st_r + z)
  combo = np.zeros([combo_r, combo_c, sum_ch], dtype=np.uint8)
  combo.fill(255)
  combo[0:sum_r, 0:sum_c] = np_sum
  combo[z:st_r + z, sum_c:sum_c + st_c] = np_stats
  result = np_to_pil(combo)
  return result

def np_tile_stat_img(tiles):
  """
  Generate tile scoring statistics for a list of tiles and return the result as a NumPy array image.

  Args:
    tiles: List of tiles (such as top tiles)

  Returns:
    Tile scoring statistics converted into an NumPy array image.
  """
  tt = sorted(tiles, key=lambda t: (t.r, t.c), reverse=False)
  tile_stats = "Tile Score Statistics:\n"
  count = 0
  for t in tt:
    if count > 0:
      tile_stats += "\n"
    count += 1
    tup = (t.r, t.c, t.rank, t.tissue_percentage, t.color_factor, t.s_and_v_factor, t.quantity_factor, t.score)
    tile_stats += "R%03d C%03d #%003d TP:%6.2f%% CF:%4.0f SVF:%4.2f QF:%4.2f S:%0.4f" % tup
  np_stats = np_text(tile_stats, font_path=SUMMARY_TITLE_FONT_PATH, font_size=14)
  return np_stats

def np_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
            font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
  """
  Obtain a NumPy array image representation of text.

  Args:
    text: The text to convert to an image.
    w_border: Tile text width border (left and right).
    h_border: Tile text height border (top and bottom).
    font_path: Path to font.
    font_size: Size of font.
    text_color: Tile text color.
    background: Tile background color.

  Returns:
    NumPy array representing the text.
  """
  pil_img = pil_text(text, w_border, h_border, font_path, font_size,
                     text_color, background)
  np_img = np.asarray(pil_img)
  return np_img

def pil_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
             font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
  """
  Obtain a PIL image representation of text.

  Args:
    text: The text to convert to an image.
    w_border: Tile text width border (left and right).
    h_border: Tile text height border (top and bottom).
    font_path: Path to font.
    font_size: Size of font.
    text_color: Tile text color.
    background: Tile background color.

  Returns:
    PIL image representing the text.
  """

  font = ImageFont.truetype(font_path, font_size)
  x, y = ImageDraw.Draw(Image.new("RGB", (1, 1), background)).textsize(text, font)
  image = Image.new("RGB", (x + 2 * w_border, y + 2 * h_border), background)
  draw = ImageDraw.Draw(image)
  draw.text((w_border, h_border), text, text_color, font=font)
  return image

def save_tile_data(tile_summary):
  """
  Save tile data to csv file.

  Args
    tile_summary: TimeSummary object.
  """

  #time = Time()

  sep=";"

  csv = "%s Tile Summary:" % tile_summary.slide_name + "\n" + summary_stats(tile_summary)

  csv += "\n\n\nTile Num" + sep + "Row" + sep + "Column" + sep + "Tissue %" + sep + "Tissue Quantity" + sep + "Col Start" + sep + "Row Start" + sep + "Col End" + sep + "Row End" + sep + "Col Size" + sep + "Row Size" + sep + "" + \
         "Original Col Start" + sep + "Original Row Start" + sep + "Original Col End" + sep + "Original Row End" + sep + "Original Col Size" + sep + "Original Row Size" + sep + "" + \
         "Color Factor" + sep + "S and V Factor" + sep + "Quantity Factor" + sep + "Score\n"

  for t in tile_summary.tiles:
    line = "%s" + sep + "%d" + sep + "%d" + sep + "%4.2f" + sep + "%s" + sep + "%d" + sep + "%d" + sep + "%d" + sep + "%d" + sep + "%d" + sep + "%d" + sep + "%d" + sep + "%d" + sep + "%d" + sep + "%d" + sep + "%d" + sep + "%d" + sep + "%4.0f" + sep + "%4.2f" + sep + "%4.2f" + sep + "%0.4f\n"
    line = line % (
      t.tile_num, t.r, t.c, t.tissue_percentage, t.tissue_quantity().name, t.c_s, t.r_s, t.c_e, t.r_e, t.c_e - t.c_s,
      t.r_e - t.r_s, t.o_c_s, t.o_r_s, t.o_c_e, t.o_r_e, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s, t.color_factor,
      t.s_and_v_factor, t.quantity_factor, t.score)
    csv += line

  data_path = s["outdir"] + os.sep + "tiles" + os.sep + s["filename"] + "_tile_data.csv"
  csv_file = open(data_path, "w")
  csv_file.write(csv)
  csv_file.close()

  #print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Data", str(time.elapsed()), data_path))

def faded_tile_border_color(tissue_percentage):
  """
  Obtain the corresponding faded tile border color for a particular tile tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage

  Returns:
    The faded tile border color corresponding to the tile tissue percentage.
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    border_color = FADED_THRESH_COLOR
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    border_color = FADED_MEDIUM_COLOR
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    border_color = FADED_LOW_COLOR
  else:
    border_color = FADED_NONE_COLOR
  return border_color