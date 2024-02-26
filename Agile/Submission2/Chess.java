abstract class Piece {

    Player p;
    Position pos;
    /**
     * This method does not have to check the validity of the position
     */
    public void setArbitraryPosition(Position currentPosition) {
        this.pos = currentPosition;
    }
    public void setArbitraryPosition(char x, int y) {
        this.pos = new Position(x,y);
    }

    /**
     * This method checks if the position is a valid position
     */
    public abstract boolean isValidPosition(Position newPosition);

    public void printPos(){
        System.out.print(pos.x + ", " + pos.y);
    }
}

class Rook extends Piece{
    public Rook (Player _p){
        p = _p;

    }

    public boolean isValidPosition(Position newPos){
        if (!newPos.inBoard()) return false;
        return pos.x == newPos.x || pos.y == newPos.y;
    }
}

class Bishop extends Piece{
    public Bishop (Player _p){
        p = _p;
    }

    public boolean isValidPosition(Position newPos){
        if (!newPos.inBoard()) return false;
        int dx = Math.abs(newPos.getPosInInt()[0] - pos.getPosInInt()[0]);
        int dy = Math.abs(newPos.getPosInInt()[1] - pos.getPosInInt()[1]);
        return dx == dy;
    }
}

class Pawn extends Piece{
    public Pawn (Player _p){
        p = _p;
    }

    public boolean isValidPosition(Position newPos){
        if (!newPos.inBoard()) return false;

        int dy = newPos.getPosInInt()[1] - pos.getPosInInt()[1];
        if(p.colorWhite){
            return pos.x == newPos.x && ((pos.getPosInInt()[1] !=2)? dy == 1:(dy == 1 || dy == 2));
        } else {
            return pos.x == newPos.x && ((pos.getPosInInt()[1] !=7)? dy == -1:(dy == -1 || dy == -2));
        }
    }
}

class King extends Piece{
    public King (Player _p){
        p = _p;
    }

    public boolean isValidPosition(Position newPos){
        if (!newPos.inBoard()) return false;
        int dx = Math.abs(newPos.getPosInInt()[0] - pos.getPosInInt()[0]);
        int dy = Math.abs(newPos.getPosInInt()[1] - pos.getPosInInt()[1]);
        return (dx <= 1) && (dy <= 1);

    }
}

class Knight extends Piece{
    public Knight (Player _p){
        p = _p;
    }

    public boolean isValidPosition(Position newPos){
        if (!newPos.inBoard()) return false;
        int dx = Math.abs(newPos.getPosInInt()[0] - pos.getPosInInt()[0]);
        int dy = Math.abs(newPos.getPosInInt()[1] - pos.getPosInInt()[1]);
        return (dx == 2 && dy == 1) || (dx == 1 && dy == 2);
    }
}




class Position{

    char x;
    int y;
    public Position(char _x, int _y){
        x = _x;
        y = _y;
    }

    public int[] getPosInInt(){
        return new int[]{((int)x-96), y};
    }

    public boolean inBoard(){
        return  (this.getPosInInt()[0]>0 && this.getPosInInt()[0]<9 && this.getPosInInt()[1]>0 && this.getPosInInt()[1]<9);
    }


}

class Player{
    String type;

    boolean colorWhite;
    public Player(String _type){
        type = _type;
    }

    public void setColorWhite(boolean b){
        colorWhite = b;
    }
}



public class Chess {

    public static void main(String[] args) {
        Player p1 = new Player("White player");
        p1.setColorWhite(true);
        Player p2 = new Player("Black player");
        p2.setColorWhite(false);

        System.out.println("Testing kings:");
        Piece whiteKing = new King(p1);
        whiteKing.setArbitraryPosition(new Position('f', 5));
        test(whiteKing, 'a', 1, false);
        test(whiteKing, 'f', 4, true);

        System.out.println("Testing rooks:");
        Rook blackRook = new Rook(p2);
        blackRook.setArbitraryPosition('d', 5);
        test(blackRook, 'h', 5, true);
        test(blackRook, 'h', 1, false);
        test(blackRook, 'd', 9, false);

        System.out.println("Testing bishops:");
        Piece whiteBishop = new Bishop(p1);
        whiteBishop.setArbitraryPosition(new Position('d', 5));
        test(whiteBishop, 'b', 2, false);
        test(whiteBishop, 'a', 8, true);

        System.out.println("Testing knigts:");
        Knight blackKnight = new Knight(p2);
        blackKnight.setArbitraryPosition('d', 4);
        test(blackKnight, 'e', 6, true);
        test(blackKnight, 'f', 6, false);
        test(blackKnight, 'c', 2, true);
        test(blackKnight, 'i', 8, false);

        System.out.println("Testing pawns:");
        Pawn whitePawn = new Pawn(p1);
        Pawn blackPawn = new Pawn(p2);
        blackPawn.setArbitraryPosition('b', 4);
        test(blackPawn, 'b', 3, true);
        test(blackPawn, 'b', 5, false);
        whitePawn.setArbitraryPosition('f', 2);
        test(whitePawn, 'f', 3, true);
        test(whitePawn, 'f', 4, true);
        blackPawn.setArbitraryPosition('g', 5);
        test(blackPawn, 'g', 4, true);
        test(blackPawn, 'g', 3, false);
        whitePawn.setArbitraryPosition('e', 7);
        test(whitePawn, 'd', 8, false);
        test(whitePawn, 'f', 8, false);
    }

    public static void test(Piece p, char x, int y, boolean valid) {
        if (p.isValidPosition(new Position(x, y)) == valid) {
            System.out.println("  > Test passed!");
        } else {
            System.out.println("  X Test NOT passed!");
        }
    }
}