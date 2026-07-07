"""Local admin UI redirect router."""

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/admin", include_in_schema=False)
async def admin_page():
    return RedirectResponse(url="/web/index.html")
