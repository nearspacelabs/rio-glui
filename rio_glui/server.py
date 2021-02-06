"""rio_glui.server: tornado tile server and template renderer."""

import os
import logging
from io import BytesIO
from concurrent import futures

import numpy

from rio_tiler.colormap import cmap
from rio_tiler.profiles import img_profiles
from rio_tiler.utils import linear_rescale, render
from rio_color.operations import parse_operations
from rio_color.utils import scale_dtype, to_math_type

from tornado import gen, web
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.concurrent import run_on_executor

from .raster import RasterTiles

logger = logging.getLogger(__name__)


class TileServer(object):
    """
    Creates a very minimal slippy map tile server using tornado.ioloop.

    Attributes
    ----------
    raster : RasterTiles
        Rastertiles object.
    tiles_format : str, optional
        Tile image format.
    scale : tuple, optional
        Min and Max data bounds to rescale data from.
        Must be in the form of "((min, max), (min, max), (min, max))" or "((min, max),)"
    colormap: str, optional
        rio-tiler compatible colormap name ('cfastie' or 'schwarzwald')
    gl_tiles_size, int, optional
        Tile pixel size. (only for templates)
    gl_tiles_minzoom: int, optional (default: 0)
        Raster tile minimun zoom. (only for templates)
    gl_tiles_maxzoom, int, optional (default: 22)
        Raster tile maximun zoom. (only for  templates)
    port, int, optional (default: 8080)
        Tornado app default port.


    Methods
    -------
    get_tiles_url()
        Get tiles endpoint url.
    get_template_url()
        Get simple app template url.
    get_bounds()
        Get raster bounds
    get_center()
        Get raster center
    get_playground_url()
        Get playground app template url.
    start()
        Start tile server.
    stop()
        Stop tile server.

    """

    def __init__(
        self,
        raster=None,
        rasters=None,
        scale=None,
        colormap=None,
        tiles_format="png",
        gl_tiles_size=None,
        gl_tiles_minzoom=0,
        gl_tiles_maxzoom=22,
        port=8080,
    ):
        """Initialize Tornado app."""
        self.raster = raster if raster else next(iter(rasters.values()))
        self.rasters = rasters if rasters else dict()
        self.port = port
        self.server = None
        self.tiles_format = tiles_format
        self.gl_tiles_size = gl_tiles_size if gl_tiles_size else self.raster.tiles_size
        self.gl_tiles_minzoom = gl_tiles_minzoom
        self.gl_tiles_maxzoom = gl_tiles_maxzoom

        settings = {"static_path": os.path.join(os.path.dirname(__file__), "static")}

        if colormap:
            colormap = cmap.get(name=colormap)

        tile_params = dict(raster=self.raster, scale=scale, colormap=colormap)
        local_tile_params = dict(rasters=rasters, scale=scale, colormap=colormap)

        template_params = dict(
            tiles_url=self.get_tiles_url(),
            tiles_bounds=self.raster.get_bounds(),
            gl_tiles_size=self.gl_tiles_size,
            gl_tiles_minzoom=self.gl_tiles_minzoom,
            gl_tiles_maxzoom=self.gl_tiles_maxzoom,
        )

        self.app = web.Application(
            [
                (r"^/tiles/(\d+)/(\d+)/(\d+)\.(\w+)", RasterTileHandler, tile_params),
                (r"^/localtiles/(\w+)/(\d+)/(\d+)/(\d+)\.(\w+)", MultiRasterTileHandler, local_tile_params),
                (r"^/index.html", IndexTemplate, template_params),
                (r"^/playground.html", PlaygroundTemplate, template_params),
                (r"/.*", InvalidAddress),
            ],
            **settings
        )

    def get_tiles_url(self):
        """Get tiles endpoint url."""
        tileformat = "jpg" if self.tiles_format == "jpeg" else self.tiles_format
        return "http://127.0.0.1:{}/tiles/{{z}}/{{x}}/{{y}}.{}".format(
            self.port, tileformat
        )

    def get_local_tiles_url(self, geotiff_name):
        tileformat = "jpg" if self.tiles_format == "jpeg" else self.tiles_format
        return "http://127.0.0.1:{}/localtiles/{}/{{z}}/{{x}}/{{y}}.{}".format(
            self.port, geotiff_name, tileformat
        )

    def get_template_url(self):
        """Get simple app template url."""
        return "http://127.0.0.1:{}/index.html".format(self.port)

    def get_playground_url(self):
        """Get playground app template url."""
        return "http://127.0.0.1:{}/playground.html".format(self.port)

    def get_bounds(self, geotiff_name=None):
        """Get RasterTiles bounds."""
        raster = self.raster if geotiff_name is None else self.rasters.get(geotiff_name)
        return raster.get_bounds()

    def get_center(self, geotiff_name=None):
        """Get RasterTiles center."""
        raster = self.raster if geotiff_name is None else self.rasters.get(geotiff_name)
        return raster.get_center()

    def start(self):
        """Start tile server."""
        is_running = IOLoop.current() is not None
        self.server = HTTPServer(self.app)
        self.server.listen(self.port)

        # NOTE: Check if there is already one server in place
        # else initiate an new one
        # When using rio-glui.server.TileServer inside
        # jupyter Notebook IOLoop is already initialized
        if not is_running:
            IOLoop.current().start()

    def stop(self):
        """Stop tile server."""
        if self.server:
            self.server.stop()


class InvalidAddress(web.RequestHandler):
    """Invalid web requests handler."""

    def get(self):
        """Retunrs 404 error."""
        raise web.HTTPError(404)


class RasterTileHandler(web.RequestHandler):
    """
    RasterTiles requests handler.

    Attributes
    ----------
    raster : RasterTiles
        Rastertiles object.

    Methods
    -------
    initialize()
        Initialize tiles handler.
    get()
        Get tile data and mask.

    """

    executor = futures.ThreadPoolExecutor(max_workers=16)

    def initialize(self, raster, scale=None, colormap=None):
        """Initialize tiles handler."""
        self.raster = raster
        self.scale = scale
        self.colormap = colormap

    @staticmethod
    def apply_color_operations(img, color_ops):
        for ops in parse_operations(color_ops):
            img = scale_dtype(ops(to_math_type(img)), numpy.uint8)

        return img

    @run_on_executor
    def _get_tile(self, z, x, y, tileformat, color_ops=None):
        if tileformat == "jpg":
            tileformat = "jpeg"

        if not self.raster.tile_exists(z, x, y):
            raise web.HTTPError(404)

        data, mask = self.raster.read_tile(z, x, y)

        if len(data.shape) == 2:
            data = numpy.expand_dims(data, axis=0)

        if self.scale:
            nbands = data.shape[0]
            scale = self.scale
            if len(scale) != nbands:
                scale = scale * nbands

            for bdx in range(nbands):
                data[bdx] = numpy.where(
                    mask,
                    linear_rescale(data[bdx], in_range=scale[bdx], out_range=(0, 255)),
                    0,
                )

            data = data.astype(numpy.uint8)

        if color_ops:
            data = RasterTileHandler.apply_color_operations(data, color_ops)

        options = img_profiles.get(tileformat, {})

        return BytesIO(
            render(
                data,
                mask=mask,
                color_map=self.colormap,
                img_format=tileformat,
                **options
            )
        )

    @gen.coroutine
    def get(self, z, x, y, tileformat):
        """Retunrs tile data and header."""
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET")
        self.set_header("Content-Type", "image/{}".format(tileformat))
        self.set_header("Cache-Control", "no-store, no-cache, must-revalidate")
        color_ops = self.get_argument("color", None)

        res = yield self._get_tile(
            int(z), int(x), int(y), tileformat, color_ops=color_ops
        )
        self.write(res.getvalue())


class MultiRasterTileHandler(RasterTileHandler):
    executor = futures.ThreadPoolExecutor(max_workers=16)

    def initialize(self, rasters, scale=None, colormap=None):
        """Initialize tiles handler."""
        self.rasters = rasters
        self.scale = scale
        self.colormap = colormap

    @gen.coroutine
    def get(self, geotiff_name, z, x, y, tileformat):
        raster = self.rasters.get(geotiff_name, None)
        if raster is None:
            raise web.HTTPError(404)
        super().initialize(raster, self.scale, self.colormap)

        yield super().get(z, x, y, tileformat)


class Template(web.RequestHandler):
    """Template requests handler.

    Attributes
    ----------
    tiles_url : str
        Tiles endpoint url.
    tiles_bounds : tuple, list
        Tiles source bounds [maxlng, maxlat, minlng, minlat].
    gl_tiles_size: int
        Tiles pixel size.
    gl_tiles_minzoom : int
        Tiles source minimun zoom level.
    gl_tiles_maxzoom : int
        Tiles source maximum zoom level.

    Methods
    -------
    initialize()
        Initialize template handler.

    """

    def initialize(
        self, tiles_url, tiles_bounds, gl_tiles_size, gl_tiles_minzoom, gl_tiles_maxzoom
    ):
        """Initialize template handler."""
        self.tiles_url = tiles_url
        self.tiles_bounds = tiles_bounds
        self.gl_tiles_size = gl_tiles_size
        self.gl_tiles_minzoom = gl_tiles_minzoom
        self.gl_tiles_maxzoom = gl_tiles_maxzoom


class IndexTemplate(Template):
    """Index template."""

    def get(self):
        """Get template."""
        params = dict(
            tiles_url=self.tiles_url,
            tiles_bounds=self.tiles_bounds,
            gl_tiles_size=self.gl_tiles_size,
            gl_tiles_minzoom=self.gl_tiles_minzoom,
            gl_tiles_maxzoom=self.gl_tiles_maxzoom,
        )

        self.render("templates/index.html", **params)


class PlaygroundTemplate(Template):
    """Playground template."""

    def get(self):
        """Get template."""
        params = dict(
            tiles_url=self.tiles_url,
            tiles_bounds=self.tiles_bounds,
            gl_tiles_size=self.gl_tiles_size,
            gl_tiles_minzoom=self.gl_tiles_minzoom,
            gl_tiles_maxzoom=self.gl_tiles_maxzoom,
        )

        self.render("templates/playground.html", **params)
