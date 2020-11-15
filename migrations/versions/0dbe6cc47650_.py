"""empty message

Revision ID: 0dbe6cc47650
Revises: 
Create Date: 2020-11-15 16:57:23.170818

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0dbe6cc47650'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('rtg_data',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('date', sa.DateTime(), nullable=True),
    sa.Column('rtg_arr', sa.ARRAY(sa.FLOAT(), dimensions=4), nullable=True),
    sa.Column('pred', sa.Numeric(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.drop_table('pneumonia_data')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('pneumonia_data',
    sa.Column('id_check', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('date_check', sa.DATE(), autoincrement=False, nullable=False),
    sa.Column('rtg_scan', postgresql.ARRAY(sa.NUMERIC()), autoincrement=False, nullable=False),
    sa.Column('pred', sa.NUMERIC(), autoincrement=False, nullable=False),
    sa.PrimaryKeyConstraint('id_check', name='pneumonia_data_pkey')
    )
    op.drop_table('rtg_data')
    # ### end Alembic commands ###