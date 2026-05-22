"""Initial schema: papers and discovery_sessions tables.

Revision ID: 001
Revises: 
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "papers",
        sa.Column("paper_id", sa.String(), primary_key=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("checksum", sa.String(), nullable=False, unique=True),
        sa.Column("status", sa.String(), nullable=False, server_default="ingested"),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_papers_checksum", "papers", ["checksum"], unique=True)
    op.create_index("ix_papers_status", "papers", ["status"])

    op.create_table(
        "discovery_sessions",
        sa.Column("session_id", sa.String(), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="running"),
        sa.Column("evidence_count", sa.Integer(), server_default="0"),
        sa.Column("hypotheses_json", sa.JSON(), nullable=True),
        sa.Column("report", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_sessions_status", "discovery_sessions", ["status"])


def downgrade() -> None:
    op.drop_table("discovery_sessions")
    op.drop_table("papers")
