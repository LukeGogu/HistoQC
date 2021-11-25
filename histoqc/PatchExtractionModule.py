import logging
import os

import math
import skimage
from histoqc.BaseImage import printMaskHelper
from skimage import io, img_as_ubyte, morphology, measure
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import rank
import numpy as np

def generateScaledMaskedImage(s, params):
    
    img = s.getImgThumb(params.get("image_work_size", "1.25x"))
    mask = skimage.transform.resize(s["img_mask_use"], img[:, :, 1].shape, order=0) > 0
    masked_img = img*np.dstack([mask, mask, mask])
    io.imsave(s["outdir"] + os.sep + s["filename"] + "_scaled_mask_applied.png", img_as_ubyte(masked_img))
    
    return

def maskedImage(img, mask):
    masked_img = img*np.dstack([mask, mask, mask])
    return masked_img

def extractTissueTiles(s, params):
    
    logging.info(f"{s['filename']} - \textractTissueTiles")
    
    patch_size = int(params.get("path_size", "128"))
    tissue_low_thresh = int(params.get("tissue_low_thresh", "10"))
    tissue_high_thresh = int(params.get("tissue_high_thresh", "80"))
    max_num_tiles = int(params.get("max_num_tiles", "None"))

    img = s.getImgThumb(params.get("image_work_size", "2.5x"))
    scaled_use_mask = skimage.transform.resize(s["img_mask_use"], img[:, :, 1].shape, order=0) > 0
    masked_img = img*np.dstack([mask, mask, mask])
    
def extractTopTiles(s, params):
    logging.info(f"{s['filename']} - \textractTopTiles")
    patch_size = int(params.get("path_size", "128"))
    img = s.getImgThumb(params.get("image_work_size", "2.5x"))
    tissue_high_thresh = int(params.get("tissue_high_thresh", "80"))
    tissue_low_thresh = int(params.get("tissue_low_thresh", "10"))
    num_top_tiles = int(params.get("num_top_tiles", "None"))







TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10

ROW_TILE_SIZE = 1024
COL_TILE_SIZE = 1024
ROW_OVERLAP = 0
COL_OVERLAP = 0

NUM_TOP_TILES = 50
MAX_TILE_NUM = 1000 # -1 means all


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
  num_row_tiles = math.ceil((rows - ROW_OVERLAP)/ (ROW_TILE_SIZE - ROW_OVERLAP))
  num_col_tiles = math.ceil((cols - COL_OVERLAP)/ (COL_TILE_SIZE - ROW_OVERLAP))
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
    end_r = start_r + ROW_TILE_SIZE if (r < num_row_tiles - 1) else h
    for c in range(0, num_col_tiles):
      start_c = c * (COL_TILE_SIZE - COL_OVERLAP)
      end_c = start_c + COL_TILE_SIZE if (c < num_col_tiles - 1) else w
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
  #if save_data:
  #  save_tile_data(tile_sum)
  #generate_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  #generate_top_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  if save_top_tiles:
    for tile in tile_sum.top_tiles():
      tile.save_tile()
  if save_tiles_with_tissue:
    for tile in tile_sum.top_tiles_tissue_thresh():
        tile.save_tile()
  return tile_sum


def save_tile_as_png(tile, save=True):
  """
  Save and/or display a tile image.

  Args:
    tile: Tile object.
    save: If True, save tile image.
    display: If True, dispaly tile image.
  """
  t = tile
  np_tile = s.getImgThumb(params.get("image_work_size", "2.5x"))[t.o_r_s:t.o_r_e, t.o_c_s:t.o_c_e]

  if save:
    img_path = s["outdir"] + os.sep + s["filename"] + os.sep + "tiles" + os.sep + "tile-r%d-c%d-x%d-y%d-w%d-h%d" % (
                             t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s) + ".png"
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
  tile_extr_img = s.getImgThumb(params.get("image_work_size", "2.5x"))
  (h, w, _) = tile_extr_img.shape
  scale_factor = max(masked_img.shape)/max(tile_extr_img.shape)

  num_row_tiles, num_col_tiles = get_num_tiles(h, w)

  tile_sum = TileSummary(height=h,
                         width=w,
                         tissue_percentage=100-mask_percent(masked_img),
                         num_col_tiles=num_col_tiles,
                         num_row_tiles=num_row_tiles)

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0

  tile_indices = get_tile_indices(h, w)
  scaled_tile_indices = tuple(np.round(np.array(tile_indices) * scale_factor))
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

    score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_tile, t_p, slide_num, r, c)

    np_scaled_tile = np_tile if small_tile_in_tile else None
    tile = Tile(tile_sum, s["filename"], np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
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

  slide_name = None
  height = None
  width = None
  mask_percentage = None
  num_row_tiles = None
  num_col_tiles = None

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0

  def __init__(self, height, width, tissue_percentage, num_col_tiles, num_row_tiles):
    self.slide_name = s["filename"]
    self.height = height
    self.width = width
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
    sorted_list_thres = filter(lambda t: t.tissue_percentage >= TISSUE_HIGH_THRESH, sorted_list)
    if MAX_TILE_NUM == -1 or MAX_TILE_NUM>len(sorted_list_thres):
        i = len(sorted_list_thres)
    else: i = MAX_TILE_NUM
    top_tiles_tissue_thresh = sorted_list_thres[:i]
    return top_tiles_tissue_thresh

  def tiles_tissue_thresh(self):

    list_thresh = filter(lambda t: t.tissue_percentage >= TISSUE_HIGH_THRESH, self.tiles())
    return tilst_thresh

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
    summary_and_tiles(self.slide_num, display=True, save_summary=False, save_data=False, save_top_tiles=False)


class Tile:
  """
  Class for information about a tile.
  """

  def __init__(self, tile_summary, slide_name, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
               o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score):
    self.tile_summary = tile_summary
    self.slide_name = slide_name
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

  def save_tile(self):
    save_tile_as_png(self, save=True)

  #def display_tile(self):
  #  save_display_tile(self, save=False, display=True)

  def display_with_histograms(self):
    display_tile(self, rgb_histograms=True, hsv_histograms=True)

  def get_np_scaled_tile(self):
    return self.np_scaled_tile

  def get_pil_scaled_tile(self):
    return util.np_to_pil(self.np_scaled_tile)


def score_tile(np_tile, tissue_percent, slide_num, row, col):
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
  hsv = filter_rgb_to_hsv(rgb, display_np_info=False)
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