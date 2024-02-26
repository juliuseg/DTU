import java.util.Random;


// ...
// write here all missing classes
// Score: increment
// Car: setlives, haslives, hit
// Obstacle: setIntensity: extends: Truck, Pillar, Life
// ...

class Score {
    int score = 0;
    public void increment(){
        score+=1;
    }

    @Override
    public String toString() {
        return String.valueOf(score);
    }

}

class Car {
    int lives;

    public void setLives(int amount){
        lives = amount;
    }

    public boolean hasLives(){
        return lives > 0;
    }

    public int getLives(){
        return lives;
    }

    public void hit (Obstacle o){
        lives += o.effect();
    }


}

abstract class Obstacle {
    public int intensity;

    public void setIntensity(int _intensity){
        intensity = _intensity;
    }
    public abstract int effect();
}

class Truck extends Obstacle{
    public int effect(){
        return -intensity;
    }
}
class Pillar extends Obstacle{
    public int effect(){
        return -intensity;
    }

}
class Life extends Obstacle{
    public int effect(){
        return intensity;
    }

}



public class CarCollisionGame {
    public static void main(String[] args) {

        Random random = new Random();
        if (args.length > 0) {
            random.setSeed(Long.parseLong(args[0]));
        }
        Car c = new Car();
        c.setLives(10);

        Score s = new Score();
        while(c.hasLives()) {
            if (random.nextDouble() < .75) {
                System.out.println("Ouch! Obstacle hit!");
                Obstacle o = null;
                double r = random.nextDouble();
                if (r < 0.4) {
                    o = new Truck(); // this should decrease the number of lives
                    System.out.println("  That was a truck!");
                } else if ( r > 0.6) {
                    o = new Pillar(); // this should decrease the number of lives
                    System.out.println("  That was a pillar!");
                } else {
                    o = new Life(); // this should increase the number of lives
                    System.out.println("  That was a new life! Hurray :)");
                }
                o.setIntensity(1 + random.nextInt(3));
                c.hit(o);
                System.out.println("  Car has now " + c.getLives() + " lives");
            } else {
                System.out.println("No obstacles hit");
            }
            s.increment();
        }

        System.out.println("Game over");
        System.out.println("Final score is " + s);
    }
}